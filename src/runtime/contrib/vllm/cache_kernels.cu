#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

namespace vllm {

template<typename scalar_t>
__global__ void reshape_and_cache_kernel(
  const scalar_t* __restrict__ key,     // [num_tokens, num_heads, head_size]
  const scalar_t* __restrict__ value,   // [num_tokens, num_heads, head_size]
  scalar_t* __restrict__ key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x]
  scalar_t* __restrict__ value_cache,   // [num_blocks, num_heads, head_size, block_size]
  const int* __restrict__ slot_mapping, // [num_tokens]
  const int key_stride,
  const int value_stride,
  const int num_heads,
  const int head_size,
  const int block_size,
  const int x) {
  const int token_idx = blockIdx.x;
  const int slot_idx = slot_mapping[token_idx];
  const int block_idx = slot_idx / block_size;
  const int block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int src_key_idx = token_idx * key_stride + i;
    const int src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int tgt_key_idx = block_idx * num_heads * (head_size / x) * block_size * x
                            + head_idx * (head_size / x) * block_size * x
                            + x_idx * block_size * x
                            + block_offset * x
                            + x_offset;
    const int tgt_value_idx = block_idx * num_heads * head_size * block_size
                              + head_idx * head_size * block_size
                              + head_offset * block_size
                              + block_offset;
    key_cache[tgt_key_idx] = __ldg(&key[src_key_idx]);
    value_cache[tgt_value_idx] = __ldg(&value[src_value_idx]);
  }
}

} // namespace vllm

namespace tvm {
namespace runtime {

TVM_REGISTER_GLOBAL("tvm.contrib.vllm.reshape_and_cache")
    .set_body_typed([](const DLTensor* key, const DLTensor* value, DLTensor* key_cache,
		       DLTensor* value_cache, const DLTensor* slot_mapping) {
      int num_tokens = key->shape[0];
      int num_heads = key->shape[1];
      int head_size = key->shape[2];
      int block_size = key_cache->shape[3];
      int vec_size = key_cache->shape[4];

      int key_stride = key->shape[1] * key->shape[2];
      int value_stride = value->shape[1] * value->shape[2];

      dim3 grid(num_tokens);
      dim3 block(std::min(num_heads * head_size, 512));

      using scalar_t = uint16_t;
      vllm::reshape_and_cache_kernel<scalar_t><<<grid, block>>>(
	static_cast<const scalar_t*>(key->data),
	static_cast<const scalar_t*>(value->data),
	static_cast<scalar_t*>(key_cache->data),
	static_cast<scalar_t*>(value_cache->data),
	static_cast<const int*>(slot_mapping->data),
        key_stride,
        value_stride,
        num_heads,
        head_size,
        block_size,
        vec_size);
    });

}  // namespace runtime
}  // namespace tvm
