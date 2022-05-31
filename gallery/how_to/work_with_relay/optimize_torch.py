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
"""
Compile PyTorch Models
======================
**Author**: `Yaoda Zhou <https://github.com/juda/>`_
This article is an introductory tutorial to optimize PyTorch models by MetaSchedule.
For us to follow this tutorial, PyTorch as well as TorchVision should be installed.
For avoiding potential "undefined symbol" issue, we strongly recommend to install PyTorch built with Cxx11 ABI from Conda, as
.. code-block:: bash
    conda install -c conda-forge pytorch-gpu
"""
# Import Tvm and PyTorch
import tvm
import torch

# Import `optimize_torch` function
from tvm.contrib.torch import optimize_torch
from torchvision.models import resnet18
from tvm.meta_schedule import TuneConfig

# Import library for profiling
import torch.utils.benchmark as benchmark

######################################################################
# Define the tuning configuration
# -------------------------------
config = TuneConfig(
                strategy="evolutionary",
                num_trials_per_iter=4,
                max_trials_per_task=4,
                max_trials_global=4,
                search_strategy_config={
                    "genetic_num_iters": 10,
                },
            )

######################################################################
# Define the resnet18 optimized by MetaSchedule
# ------------------------------
# For learning how to define a resnet18 model via PyTorch's nn.Module, 
# you can refer to https://pytorch.org/docs/stable/jit.html#mixing-tracing-and-scripting

# In our working machine, the GPU model is nvidia/geforce-rtx-3070.
dev = tvm.cuda(0)
tar_cuda = "nvidia/geforce-rtx-3070"

# For PyTorch users, you can write your nn.Module in a normal way.
# By applying "optimize_torch" function on the resnet18 model, we obtain a new resnet18 model optimized by MetaSchedule
class MyScriptModule(torch.nn.Module):
    def __init__(self, config, device = None, target = None):
        super(MyScriptModule, self).__init__()
        self.means = torch.nn.Parameter(torch.tensor([103.939, 116.779, 123.68])
                                        .resize_(1, 3, 1, 1)).cuda()
        self.resnet = optimize_torch(resnet18(), [torch.rand(1, 3, 224, 224)], config, device, target)

    def forward(self, input):
        return self.resnet(input - self.means)

# If we set the last two parameters (device and target) empty, the model will deploy on CPU.
# Since the setting of the number of trials is large, the initialization could be slow (sometimes more than 3 hours!)
my_script_module = MyScriptModule(config, dev, tar_cuda)


######################################################################
# Define the resnet18 optimized by TorchScript
# ------------------------------
# Besides, let us define a resnet18 model in a standard way.
# TorchScript also provide a built-in "optimize_for_inference" function to accelerate the inference.

class JitModule(torch.nn.Module):
    def __init__(self):
        super(JitModule, self).__init__()
        self.means = torch.nn.Parameter(torch.tensor([103.939, 116.779, 123.68])
                                        .resize_(1, 3, 1, 1)).cuda()
        self.resnet = torch.jit.optimize_for_inference(torch.jit.script(resnet18().cuda().eval()))
        
    def forward(self, input):
        return self.resnet(input - self.means)
    
jit_module = JitModule()

######################################################################
# Compare the performance between two scheduling approaches.
# ------------------------------
# Using PyTorch's benchmark Compare class, we can have a straightforward comparison between two inference models.

results = []
for i in range(20):
    test_input = torch.rand(1, 3, 224, 224).half().cuda()
    sub_label = f'[test {i}]'
    results.append(benchmark.Timer(
            stmt='my_script_module(test_input)',
            setup='from __main__ import my_script_module',
            globals={'test_input': test_input},
            sub_label=sub_label,
            description='tuning by meta',
        ).blocked_autorange())
    results.append(benchmark.Timer(
            stmt='jit_module(test_input)',
            setup='from __main__ import jit_module',
            globals={'test_input': test_input},
            sub_label=sub_label,
            description='tuning by jit',
        ).blocked_autorange())

# We can print the results on screen.
compare = benchmark.Compare(results)
compare.print()