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

import tvm


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model",
        type=str,
        required=True,
        help="""
        The name of model to run. It should a directory name under 
        https://github.com/pytorch/benchmark/tree/main/torchbenchmark/models.
        """,
    )
    args.add_argument(
        "--frontend",
        type=str,
        default="torchdynamo",
        choices=["torchdynamo"],
        help="The frontend to ingest model from torchbench.",
    )
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
        "--num-trials",
        type=int,
        required=True,
        help="The number of trials to run per iteration of MetaSchedule.",
    )
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
ARGS = _parse_args()


def main():
    pass


if __name__ == "__main__":
    main()
