import torch

import tvm
import tvm.testing
from tvm.contrib.torch import optimize_torch
from tvm.meta_schedule import TuneConfig


def negate(x):
    return x.logical_not()


optimized_negate = optimize_torch(negate, torch.ones(1, dtype=torch.bool), tuning_config=TuneConfig(
                strategy="evolutionary",
                num_trials_per_iter=4,
                max_trials_per_task=8,
                max_trials_global=0,
            ))


def test_bool_tensor_copying():
    input = torch.ones(1, dtype=torch.bool)
    output = optimized_negate(negate(input))
    print(input.equal(output))


if __name__ == "__main__":
    test_bool_tensor_copying()
