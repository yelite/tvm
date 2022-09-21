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
import logging
import os
import sys

import tvm
from tvm.support import describe
from tvm.meta_schedule.testing.torchbench.util import load_torchbench, load_torchdynamo

torchdynamo = load_torchdynamo()


def parse_args():
    args = argparse.ArgumentParser()

    args.add_argument(
        "--no-tuning",
        action="store_true",
        help="""
        Run the script without tuning. It compiles the model using
        existing database at workdir.
        """,
    )
    args.add_argument(
        "--no-benchmark",
        action="store_true",
        help="""
        Skip running the benchmark and exit right after tuning finishes.
        """,
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


def get_benchmark_runner():
    pass


def get_torchdynamo_backend():
    pass


def run_benchmark():
    pass


def main():
    describe()

    if not ARGS.without_tuning:
        pass

    if ARGS.only_tuning:
        pass

    


if __name__ == "__main__":
    main()

