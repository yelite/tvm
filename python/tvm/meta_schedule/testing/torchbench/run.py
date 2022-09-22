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
import argparse
import functools
import logging

import numpy as np
import torch
from scipy.stats import ttest_ind

import tvm
from tvm import meta_schedule as ms
from tvm.contrib import graph_executor
from tvm.meta_schedule.testing.torchbench.util import (
    load_torchdynamo_benchmark_runner, same, timed)
from tvm.support import describe

runner = load_torchdynamo_benchmark_runner()
import torchdynamo


def parse_args():
    args = argparse.ArgumentParser()

    args.add_argument(
        "--disable-tuning",
        action="store_true",
        help="""
        Run the script without tuning. It compiles the model using
        existing database at workdir.
        """,
    )
    args.add_argument(
        "--disable-benchmark",
        action="store_true",
        help="""
        Skip running the benchmark and exit right after tuning finishes.
        """,
    )
    args.add_argument(
        "--benchmark-repeat",
        type=int,
        default=30,
        help="The number of times to repeat the benchmark measurement.",
    )

    # Model selection
    args.add_argument(
        "--model",
        type=str,
        required=True,
        help="""
        The name of model to run. It should a directory name under 
        https://github.com/pytorch/benchmark/tree/main/torchbenchmark/models.
        """,
    )

    # Tuning-related config
    args.add_argument(
        "--target",
        type=tvm.target.Target,
        required=True,
        help="The target to tune and run benchmark for.",
    )
    args.add_argument(
        "--work-dir",
        type=str,
        required=True,
        help="The working directory to save intermediate results.",
    )
    args.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="""
        The directory to cache the generated network.
        If not specified, the cache will be disabled.
        """,
    )
    args.add_argument(
        "--num-trials",
        type=int,
        required=True,
        help="The number of trials to run per iteration of MetaSchedule.",
    )
    args.add_argument(
        "--backend",
        type=str,
        choices=["graph", "vm"],
        default="graph",
        help="The backend to use for relay compilation(graph / vm).",
    )
    # TODO: Add a layout arg to transform the network after
    #       ingesting into Relay and before feeding into MetaSchedule.

    # Evaluator-related config
    args.add_argument(
        "--number",
        type=int,
        default=3,
        help="The number of times to run the model for taking average in a single measurement.",
    )
    args.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="The number of times to repeat the measurement.",
    )
    args.add_argument(
        "--min-repeat-ms",
        type=int,
        default=100,
        help="""
        Minimum repeat time in ms. The number of runs will be increased if the actual
        repeat time is lowered than this.
        """,
    )
    args.add_argument(
        "--disable-adaptive-training",
        action="store_true",
        help="Whether to disable adpative training for cost model.",
    )
    args.add_argument(
        "--disable-cpu-flush",
        action="store_true",
        help="Whether to disable CPU cache flush.",
    )

    # RPC-related args
    args.add_argument(
        "--rpc-host",
        type=str,
        help="Host of the RPC Tracker for tuning. Use LocalRunner if not provided",
    )
    args.add_argument(
        "--rpc-port",
        type=int,
        help="Port of the RPC Tracker for tuning",
    )
    args.add_argument(
        "--rpc-key",
        type=str,
        help="Key of the RPC Tracker for tuning",
    )

    parsed = args.parse_args()
    return parsed


logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("tvm.meta_schedule").setLevel(logging.INFO)
ARGS = parse_args()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_tvm_backend():
    def backend(graph_module, example_inputs):
        jit_mod = torch.jit.trace(graph_module, example_inputs)
        shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(example_inputs)]
        ir_mod, params = tvm.relay.frontend.from_pytorch(jit_mod, shape_list)

        lib = ms.tune_relay(
            mod=ir_mod,
            target=ARGS.target,
            config=ms.TuneConfig(
                strategy="evolutionary",
                num_trials_per_iter=64,
                max_trials_per_task=ARGS.num_trials,
                max_trials_global=ARGS.num_trials,
                adaptive_training=not ARGS.disable_adaptive_training,
            ),
            work_dir=ARGS.work_dir,
            params=params,
            backend=ARGS.backend,
        )
        
        device = tvm.cuda(0) if ARGS.target.kind.name == "cuda" else tvm.cpu(0)

        if ARGS.backend == "graph":
            mod = graph_executor.GraphModule(lib["default"](device))
        elif ARGS.backend == "vm":
            raise RuntimeError("vm backend not supported yet") 

        # From https://github.com/pytorch/torchdynamo/blob/main/torchdynamo/optimizations/backends.py#L712
        def forward(*args):
            args = [arg.contiguous() for arg in args]
            for idx, arg in enumerate(args, 0):
                mod.set_input(
                    f"inp_{idx}",
                    tvm.nd.from_dlpack(arg),
                )
            mod.run()
            return [torch.from_dlpack(mod.get_output(i)) for i in range(mod.get_num_outputs())]

        return forward

    return backend


def performance_experiment(model_iter_fn, model, example_inputs):
    # Simplified from https://github.com/pytorch/torchdynamo/blob/c537639f9712621dc04ca09908796dbbe86c354b/benchmarks/common.py#L494

    timings = np.zeros((ARGS.benchmark_repeat, 2), np.float64)

    is_correct = True

    frozen_model_iter_fn = torchdynamo.run(model_iter_fn)
    for rep in range(ARGS.benchmark_repeat):
        # interleave the runs to handle frequency scaling and load changes
        timings[rep, 0], expected_output = timed(
            model, model_iter_fn, example_inputs, return_result=True
        )
        timings[rep, 1], actual_output = timed(
            model, frozen_model_iter_fn, example_inputs, return_result=True
        )
        is_correct = is_correct and same(expected_output, actual_output)

    pvalue = ttest_ind(timings[:, 0], timings[:, 1]).pvalue
    median = np.median(timings, axis=0)
    speedup = median[0] / median[1]
    logger.info(
        f"eager:{median[0]:.3g} optimized:{median[1]:.3g} speedup:{speedup:.3f}x p:{pvalue:.3f}"
    )

    return ""


def main():
    describe()

    optimize_ctx = torchdynamo.optimize(create_tvm_backend())

    try:
        device, name, model, example_inputs, batch_size = runner.load_model(
            ARGS.target.kind.name,
            ARGS.model,
        )
    except NotImplementedError as e:
        logging.exception(f"{ARGS.model} failed to load")
        return

    experiment = functools.partial(performance_experiment, runner.model_iter_fn)
    runner.run_one_model(name, model, example_inputs, optimize_ctx, experiment)


if __name__ == "__main__":
    main()
