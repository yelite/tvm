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


test_cache()
