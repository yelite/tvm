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
import os
import sys


def find_library_dir(name, candidates):
    for library_dir in candidates:
        if os.path.exists(library_dir):
            break

    assert os.path.exists(library_dir), f"Cannot find dir for {name}"
    return library_dir


def load_torchbench():
    try:
        import torchbenchmark
    except ImportError:
        sys.path.append(
            find_library_dir(
                "torchbench",
                [
                    "../benchmark",
                    "../../benchmark",
                ],
            )
        )
        import torchbenchmark

    return torchbenchmark


def load_torchdynamo():
    try:
        import torchdynamo
    except ImportError:
        sys.path.append(
            find_library_dir(
                "torchdynamo",
                [
                    "../torchdynamo",
                    "../../torchdynamo",
                ],
            )
        )
        import torchdynamo

    return torchdynamo
