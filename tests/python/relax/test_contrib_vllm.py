import numpy as np

import torch

import tvm
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


def to_torch(arr):
    return torch.from_numpy(arr).to("cuda")


def test_attention():
    @I.ir_module
    class Module:
        @R.function
        def main(
            query: R.Tensor(("num_seqs", 128, 32), dtype="float16"),
            key_cache: R.Tensor(("num_blocks", 128, 2, 16, 16), dtype="float16"),
            value_cache: R.Tensor(("num_tokens", 128, 32, 16), dtype="float16"),
            head_mapping: R.Tensor((128,), dtype="int32"),
            block_tables: R.Tensor(("num_seqs", 128), dtype="int32"),
            context_lens: R.Tensor(("num_seqs",), dtype="int32"),
        ) -> R.Tuple(
            [
                R.Tensor(("num_blocks", 128, 2, 16, 16), dtype="float16"),
                R.Tensor(("num_tokens", 128, 32, 16), dtype="float16"),
            ]
        ):
            with R.dataflow():
                max_len = R.max(context_lens)
                out = R.call_dps_packed(
                    "tvm.contrib.vllm.single_query_cached_kv_attention",
                    [
                        query,
                        key_cache,
                        value_cache,
                        head_mapping,
                        block_tables,
                        context_lens,
                        16,
                        max_len,
                    ],
                    out_sinfo=query.struct_info,
                )
                R.output(out)
            return out

    print(Module)
    return

    scale = 0.125
    block_size = 16
    max_context_len = 21

    query = np.load("vllm_attention_inputs/query.npy")
    key_cache = np.load("vllm_attention_inputs/key_cache.npy")
    value_cache = np.load("vllm_attention_inputs/value_cache.npy")
    block_tables = np.load("vllm_attention_inputs/block_tables.npy")
    head_mapping = np.load("vllm_attention_inputs/head_mapping.npy")
    context_lens = np.load("vllm_attention_inputs/context_lens.npy")

    ref = np.load("vllm_attention_inputs/output.npy")

    output = to_torch(np.zeros_like(ref))

    from vllm import attention_ops

    attention_ops.single_query_cached_kv_attention(
        output,
        to_torch(query),
        to_torch(key_cache),
        to_torch(value_cache),
        to_torch(head_mapping),
        scale,
        to_torch(block_tables),
        to_torch(context_lens),
        block_size,
        max_context_len,
        None,  # alibi_slopes
    )

    print(np.max(np.abs(ref - output.cpu().numpy())))


def test_cache():
    @I.ir_module
    class Module:
        @R.function
        def main(
            key: R.Tensor(("num_tokens", 128, 32), dtype="float16"),
            value: R.Tensor(("num_tokens", 128, 32), dtype="float16"),
            key_cache: R.Tensor(("num_blocks", 128, 2, 16, 16), dtype="float16"),
            value_cache: R.Tensor(("num_blocks", 128, 32, 16), dtype="float16"),
            slot_mapping: R.Tensor(("num_tokens",), dtype="int32"),
        ) -> R.Tuple(
            [
                R.Tensor(("num_blocks", 128, 2, 16, 16), dtype="float16"),
                R.Tensor(("num_blocks", 128, 32, 16), dtype="float16"),
            ]
        ):
            _ = R.call_packed(
                "tvm.contrib.vllm.reshape_and_cache",
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                sinfo_args=[R.Tuple()],
            )
            with R.dataflow():
                out = (key_cache, value_cache)
                R.output(out)
            return out

    print(Module)
    return

    key_to_cache = to_torch(np.load("vllm_cache_inputs/key_to_cache.npy"))
    value_to_cache = to_torch(np.load("vllm_cache_inputs/value_to_cache.npy"))

    key_cache = to_torch(np.load("vllm_cache_inputs/key_cache_before.npy"))
    value_cache = to_torch(np.load("vllm_cache_inputs/value_cache_before.npy"))
    slot_mapping = to_torch(np.load("vllm_cache_inputs/slot_mapping.npy"))

    from vllm import cache_ops

    cache_ops.reshape_and_cache(
        key_to_cache,
        value_to_cache,
        key_cache,
        value_cache,
        slot_mapping,
    )

    key_cache_after = np.load("vllm_cache_inputs/key_cache_after.npy")
    value_cache_after = np.load("vllm_cache_inputs/value_cache_after.npy")

    print(np.max(np.abs(key_cache_after - key_cache.cpu().numpy())))
    print(np.max(np.abs(value_cache_after - value_cache.cpu().numpy())))


test_attention()
