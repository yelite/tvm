# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import torch

try:
    from flash_attn import flash_attn_with_kvcache

    has_flash_attn_py = True
except:
    has_flash_attn_py = False

import numpy as np
import pytest

import tvm.testing
import tvm
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


has_flash = tvm.get_global_func("tvm.contrib.flash_attn.flash_decoding_with_paged_kvcache", True)

flash_attn_enabled = pytest.mark.skipif(
    not has_flash,
    reason="Flash attention not enabled.",
)

pytestmark = [flash_attn_enabled]


def test_flash_decoding_with_paged_kvcache():
    @I.ir_module
    class Module:
        @R.function
        def main(
            query: R.Tensor(("num_tokens", 40, 128), dtype="float16"),
            key_cache: R.Tensor(("num_blocks", 256, 40, 128), dtype="float16"),
            value_cache: R.Tensor(("num_blocks", 256, 40, 128), dtype="float16"),
            block_tables: R.Tensor(("num_seqs", "max_num_blocks_per_seq"), dtype="int32"),
            context_lens: R.Tensor(("num_seqs",), dtype="int32"),
        ) -> R.Tensor(("num_tokens", 40, 128), dtype="float16"):
            with R.dataflow():
                num_seqs = T.int64()
                num_tokens = T.int64()
                seqlen_q = T.floordiv(num_tokens, num_seqs)

                query_reshaped = R.reshape(query, [num_seqs, -1, 40, 128])

                # alloc workspace
                softmax_lse_accum = R.zeros((128, num_seqs, 40, seqlen_q), "float32")
                output_accum = R.zeros((128, num_seqs, 40, seqlen_q, 128), "float32")

                out = R.call_dps_packed(
                    "tvm.contrib.flash_attn.flash_decoding_with_paged_kvcache",
                    [
                        query_reshaped,
                        key_cache,
                        value_cache,
                        block_tables,
                        context_lens,
                        softmax_lse_accum,
                        output_accum,
                    ],
                    out_sinfo=query_reshaped.struct_info,
                )

                out2 = R.reshape(out, [num_tokens, 40, 128])

                R.output(out2)

            return out2

    np.random.seed(0)
    num_heads = 40
    head_dim = 128
    block_size = 256
    num_seqs = 2
    num_blocks = 210

    target = "cuda"

    with tvm.target.Target(target):
        mod = relax.transform.LegalizeOps()(Module)
        mod = tvm.tir.transform.DefaultGPUSchedule()(mod)

    with tvm.transform.PassContext():
        ex = relax.build(mod, target)

    key_cache = np.random.randn(num_blocks, block_size, num_heads, head_dim).astype("float16")
    value_cache = np.random.randn(num_blocks, block_size, num_heads, head_dim).astype("float16")

    for seqlen_q in [1, 5]:
        query = np.random.randn(num_seqs * seqlen_q, num_heads, head_dim).astype("float16")
        block_tables = np.array([[208], [209]]).astype("int32")
        context_lens = np.array([69, 129]).astype("int32")

        dev = tvm.device(target, 0)
        vm = relax.VirtualMachine(ex, dev)
        f = vm["main"]

        inputs_np = [query, key_cache, value_cache, block_tables, context_lens]
        inputs = [tvm.nd.array(inp, dev) for inp in inputs_np]

        out = f(*inputs).numpy()

        if has_flash_attn_py:

            def to_torch(arr):
                return torch.from_numpy(arr).to("cuda")

            ref = (
                flash_attn_with_kvcache(
                    to_torch(np.reshape(query, (num_seqs, seqlen_q, num_heads, head_dim))),
                    to_torch(key_cache),
                    to_torch(value_cache),
                    cache_seqlens=to_torch(context_lens),
                    block_table=to_torch(block_tables),
                    causal=True,
                )
                .cpu()
                .numpy()
            )

            ref = np.reshape(ref, [num_seqs * seqlen_q, num_heads, head_dim])

            assert np.max(np.abs(ref - out)) == 0.0


def test_reconstruct_from_cache():
    num_heads = 1
    head_dim = 8
    block_size = 256
    num_tokens = 8
    num_blocks = 1

    dev = tvm.device("cuda", 0)

    key = tvm.nd.array(np.random.randn(num_tokens, num_heads, head_dim).astype("float16"), dev)
    value = tvm.nd.array(np.random.randn(num_tokens, num_heads, head_dim).astype("float16"), dev)
    slot_mapping = tvm.nd.array(np.arange(num_tokens).astype("int32"), dev)

    k_cache = tvm.nd.array(
        np.random.randn(num_blocks, block_size, num_heads, head_dim).astype(
            "float16"
        ),
        dev,
    )
    v_cache = tvm.nd.array(
        np.random.randn(num_blocks, block_size, num_heads, head_dim).astype("float16"), dev
    )

    update_cache_func = tvm.get_global_func("tvm.contrib.flash_attn.update_cache")
    reconstruct_from_cache_func = tvm.get_global_func("tvm.contrib.flash_attn.reconstruct_from_cache")

    update_cache_func(key, value, k_cache, v_cache, slot_mapping)
    out = reconstruct_from_cache_func(k_cache, v_cache, slot_mapping)

    np.testing.assert_equal(key.numpy(), out[0].numpy())
    np.testing.assert_equal(value.numpy(), out[1].numpy())


if __name__ == "__main__":
    tvm.testing.main()
