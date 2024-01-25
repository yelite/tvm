/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/container/shape_tuple.h>

#include "../../../3rdparty/libflash_attn/include/flash.h"

namespace tvm {
namespace runtime {
namespace flash_attn {

Array<NDArray> AllocateKVCache(int head_size, int num_layers, int num_heads, int block_size,
                               int num_blocks) {
  Array<NDArray> cache;

  int device_id;
  cudaGetDevice(&device_id);

  DLDevice dev{DLDeviceType::kDLCUDA, device_id};
  ShapeTuple block_shape{num_blocks, block_size, num_heads, head_size};

  for (int i = 0; i < num_layers; ++i) {
    NDArray key_blocks = NDArray::Empty(block_shape, runtime::DataType::Float(16), dev);
    NDArray value_blocks = NDArray::Empty(block_shape, runtime::DataType::Float(16), dev);
    cache.push_back(key_blocks);
    cache.push_back(value_blocks);
  }

  return cache;
}

template <typename scalar_t>
__global__ void update_cache_kernel(
    const scalar_t* __restrict__ key,          // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,        // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key_cache,          // [num_blocks, block_size, num_heads, head_size]
    scalar_t* __restrict__ value_cache,        // [num_blocks, block_size, num_heads, head_size]
    const int* __restrict__ slot_mapping,  // [num_tokens]
    int stride, int num_heads, int head_size) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }
  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t src_idx = token_idx * stride + i;
    const int64_t tgt_idx = slot_idx * n + i;
    key_cache[tgt_idx] = key[src_idx];
    value_cache[tgt_idx] = value[src_idx];
  }
}

// Grid: (num_layers, num_pairs)
template <typename scalar_t>
__global__ void copy_blocks_kernel(int64_t* key_cache_ptrs, int64_t* value_cache_ptrs,
                                   const int64_t* __restrict__ block_mapping,
                                   const int numel_per_block) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;

  scalar_t* key_cache = reinterpret_cast<scalar_t*>(key_cache_ptrs[layer_idx]);
  scalar_t* value_cache = reinterpret_cast<scalar_t*>(value_cache_ptrs[layer_idx]);
  int64_t src_block_number = block_mapping[2 * pair_idx];
  int64_t dst_block_number = block_mapping[2 * pair_idx + 1];

  const int64_t src_block_offset = src_block_number * numel_per_block;
  const int64_t dst_block_offset = dst_block_number * numel_per_block;
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    key_cache[dst_offset] = key_cache[src_offset];
  }
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    value_cache[dst_offset] = value_cache[src_offset];
  }
}

template <typename scalar_t>
__global__ void reconstruct_from_cache_kernel(
    const scalar_t* __restrict__ key_cache,  // [num_blocks, block_size, num_heads, head_size]
    const scalar_t* __restrict__ value_cache,  // [num_blocks, block_size, num_heads, head_size]
    const int* __restrict__ slot_mapping,      // [num_tokens]
    scalar_t* __restrict__ key,                // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ value,              // [num_tokens, num_heads, head_size]
    const int stride, const int num_heads, const int head_size, const int block_size) {
  const int token_idx = blockIdx.x;
  const int slot_idx = slot_mapping[token_idx];

  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t tgt_idx = token_idx * stride + i;
    const int64_t src_idx = slot_idx * n + i;
    key[tgt_idx] = key_cache[src_idx];
    value[tgt_idx] = value_cache[src_idx];
  }

}

}  // namespace flash_attn

/*
  query: (batch_size, seqlen_q, num_heads, head_size), fp16
  key_cache: (num_blocks, page_block_size, num_heads_k, head_size), fp16
  value_cache: num_blocks, page_block_size, num_heads_k, head_size), fp16
  block_tables: (batch_size, max_num_blocks_per_seq), int32
  context_lens: (batch_size,), int32
  softmax_lse_accum: (max_num_splits, batch_size, num_heads, seqlen_q), fp32
  output_accum: (max_num_splits, batch_size, num_heads, seqlen_q, head_size), fp32
  out: (batch_size, seqlen_q, num_heads, head_size), fp16
*/
TVM_REGISTER_GLOBAL("tvm.contrib.flash_attn.flash_decoding_with_paged_kvcache")
  .set_body_typed([](const DLTensor* query, const DLTensor* key_cache,
                     const DLTensor* value_cache, const DLTensor* block_tables,
                     const DLTensor* context_lens, DLTensor* softmax_lse_accum,
                     DLTensor* output_accum, DLTensor* out) {
      int batch_size = query->shape[0];
      int seqlen_q = query->shape[1];
      int num_heads = query->shape[2];
      int head_dim = query->shape[3];
      int num_heads_k = key_cache->shape[2];
      int num_blocks = key_cache->shape[0];
      int block_size = key_cache->shape[1];
      int max_num_blocks_per_seq = block_tables->shape[1];
      float softmax_scale = 1.0 / sqrt(static_cast<float>(head_dim));

      ICHECK(block_size % 64 == 0) << "Block size needs to be a multiple of 64.";

      auto block_table_ptr = static_cast<int*>(block_tables->data);
      auto seqlens_k_ptr = static_cast<int*>(context_lens->data);

      using half = ::flash_attn::half;

      ICHECK(TypeMatch(block_tables->dtype, kDLInt, 32));
      ICHECK(TypeMatch(context_lens->dtype, kDLInt, 32));
      ICHECK(TypeMatch(softmax_lse_accum->dtype, kDLFloat, 32));
      ICHECK(TypeMatch(output_accum->dtype, kDLFloat, 32));

      auto q_ptr = static_cast<half*>(query->data);
      auto kcache_ptr = static_cast<half*>(key_cache->data);
      auto vcache_ptr = static_cast<half*>(value_cache->data);
      auto softmax_lse_accum_ptr = static_cast<float*>(softmax_lse_accum->data);
      auto output_accum_ptr = static_cast<float*>(output_accum->data);
      auto output_ptr = static_cast<half*>(out->data);

      int q_head_stride = head_dim;
      int k_head_stride = head_dim;
      int v_head_stride = head_dim;
      int o_head_stride = head_dim;
      int q_row_stride = q_head_stride * num_heads;
      int k_row_stride = k_head_stride * num_heads_k;
      int v_row_stride = v_head_stride * num_heads_k;
      int o_row_stride = o_head_stride * num_heads;
      int q_batch_stride = q_row_stride * seqlen_q;
      int k_batch_stride = k_row_stride * block_size;
      int v_batch_stride = v_row_stride * block_size;
      int o_batch_stride = o_row_stride * seqlen_q;
      int block_table_batch_stride = max_num_blocks_per_seq;

      ::flash_attn::flash_attention_splitkv_paged_forward(
          q_ptr, kcache_ptr, vcache_ptr, block_table_ptr, seqlens_k_ptr,
          softmax_lse_accum_ptr, output_accum_ptr,
          output_ptr, batch_size, seqlen_q, num_heads, num_heads_k, head_dim,
          q_batch_stride,
          k_batch_stride,
          v_batch_stride,
          o_batch_stride,
          q_head_stride,
          k_head_stride,
          v_head_stride,
          o_head_stride,
          q_row_stride,
          k_row_stride,
          v_row_stride,
          o_row_stride,
          num_blocks, block_size, max_num_blocks_per_seq,
          block_table_batch_stride,
          softmax_scale,
          true /* is_causal*/);
    });

TVM_REGISTER_GLOBAL("tvm.contrib.flash_attn.allocate_kv_cache").set_body_typed(flash_attn::AllocateKVCache);

TVM_REGISTER_GLOBAL("tvm.contrib.flash_attn.update_cache")
    .set_body_typed([](NDArray key, NDArray value, NDArray key_cache, NDArray value_cache,
                       NDArray slot_mapping) {
      int num_tokens = key->shape[0];
      int num_heads = key->shape[1];
      int head_size = key->shape[2];
      int stride = key->shape[1] * key->shape[2];

      dim3 grid(num_tokens);
      dim3 block(std::min(num_heads * head_size, 512));

      using scalar_t = uint16_t;

      flash_attn::update_cache_kernel<scalar_t><<<grid, block>>>(
          static_cast<const scalar_t*>(key->data),
	  static_cast<const scalar_t*>(value->data),
          static_cast<scalar_t*>(key_cache->data),
	  static_cast<scalar_t*>(value_cache->data),
          static_cast<const int*>(slot_mapping->data),
	  stride, num_heads, head_size);

      return Array{key_cache, value_cache};
    });

TVM_REGISTER_GLOBAL("tvm.contrib.flash_attn.copy_blocks")
    .set_body_typed([](Array<NDArray> key_value_caches, NDArray block_mapping) {
      auto num_layers = key_value_caches.size() / 2;
      auto num_pairs = block_mapping->shape[0] / 2;

      if (num_layers == 0) {
        return;
      }

      std::vector<int64_t> key_cache_ptrs(num_layers);
      std::vector<int64_t> value_cache_ptrs(num_layers);
      for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        key_cache_ptrs[layer_idx] =
            reinterpret_cast<int64_t>(key_value_caches[2 * layer_idx]->data);
        value_cache_ptrs[layer_idx] =
            reinterpret_cast<int64_t>(key_value_caches[2 * layer_idx + 1]->data);
      }

      NDArray key_cache = key_value_caches[1];  // [num_blocks, num_heads, head_size, block_size]
      DLDevice dev = key_cache->device;

      NDArray key_cache_ptrs_gpu =
          NDArray::Empty({static_cast<int>(num_layers)}, runtime::DataType::Int(64), dev);
      NDArray value_cache_ptrs_gpu =
          NDArray::Empty({static_cast<int>(num_layers)}, runtime::DataType::Int(64), dev);
      key_cache_ptrs_gpu.CopyFromBytes(key_cache_ptrs.data(),
                                       sizeof(int64_t) * key_cache_ptrs.size());
      value_cache_ptrs_gpu.CopyFromBytes(value_cache_ptrs.data(),
                                         sizeof(int64_t) * value_cache_ptrs.size());

      NDArray block_mapping_gpu =
          NDArray::Empty(block_mapping.Shape(), runtime::DataType::Int(64), dev);
      block_mapping_gpu.CopyFromBytes(block_mapping->data,
                                      sizeof(int64_t) * block_mapping->shape[0]);

      const int numel_per_block = key_cache->shape[1] * key_cache->shape[2] * key_cache->shape[3];
      dim3 grid(num_layers, num_pairs); dim3 block(std::min(1024, numel_per_block));

      using scalar_t = uint16_t;
      flash_attn::copy_blocks_kernel<scalar_t>
          <<<grid, block>>>(static_cast<int64_t*>(key_cache_ptrs_gpu->data),
                            static_cast<int64_t*>(value_cache_ptrs_gpu->data),
                            static_cast<int64_t*>(block_mapping_gpu->data), numel_per_block);
    });

TVM_REGISTER_GLOBAL("tvm.contrib.flash_attn.reconstruct_from_cache")
    .set_body_typed([](NDArray key_cache, NDArray value_cache, NDArray slot_mapping) {
      int num_tokens = slot_mapping->shape[0];
      int num_heads = value_cache->shape[2];
      int head_size = value_cache->shape[3];
      int block_size = value_cache->shape[1];

      DLDevice dev = key_cache->device;
      auto key = NDArray::Empty({num_tokens, num_heads, head_size}, key_cache->dtype, dev);
      auto value = NDArray::Empty({num_tokens, num_heads, head_size}, key_cache->dtype, dev);

      int stride = key->shape[1] * key->shape[2];

      dim3 grid(num_tokens);
      dim3 block(std::min(num_heads * head_size, 512));

      using scalar_t = uint16_t;
      flash_attn::reconstruct_from_cache_kernel<scalar_t>
          <<<grid, block>>>(static_cast<const scalar_t*>(key_cache->data),
                            static_cast<const scalar_t*>(value_cache->data),
                            static_cast<const int*>(slot_mapping->data),
                            static_cast<scalar_t*>(key->data), static_cast<scalar_t*>(value->data),
                            stride, num_heads, head_size, block_size);

      return Array{key, value};
    });

}  // namespace runtime
}  // namespace tvm
