# pylint: disable=inconsistent-return-statements
#!/usr/bin/env python

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
from typing import Callable, List, Tuple, Union

import torch
import torch.utils.dlpack

import tvm


class OperatorModuleWrapper(torch.nn.Module):
    def __init__(self, module: Union[tvm.ir.module.IRModule, tvm.tir.function.PrimFunc, tvm.contrib.graph_executor.GraphModule]):
        super().__init__()
        self.engine_cpu = None
        self.engine_cuda = None
        self.ir_module = module

    def build_cpu(self):
        runtime_module = tvm.build(self.ir_module)
        func = tvm.get_global_func("tvmtorch.save_runtime_mod")
        func(runtime_module)

        self.engine_cpu = torch.classes.tvm_torch.OperatorModuleWrapper()

    def build_cuda(self):
        # If the module build on cuda, we won't call the C++ code since some information is missing
        runtime_module = tvm.build(self.ir_module, target=tvm.target.cuda())
        self.engine_cuda = runtime_module

    def forward(self, *torch_inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        if torch_inputs[0].is_cuda:
            if self.engine_cuda is None:
                self.build_cuda()
            return self.engine_cuda.forward(torch_inputs)
        else:
            # We force the tensor inputs to be on cpu.
            torch_inputs = tuple(map(lambda x: x.cpu(), torch_inputs))
            if self.engine_cpu is None:
                self.build_cpu()
            return self.engine_cpu.forward(torch_inputs)


def as_torch(
    func: Union[tvm.ir.module.IRModule, tvm.tir.function.PrimFunc, Callable]
):
    """A decorator of converting TensorIR to PyTorch nn.Module.

    Parameters
    ----------
    func : Union[tvm.ir.module.IRModule, tvm.tir.function.PrimFunc, Callable]
        The function to be parsed.


    Returns
    -------
    mod : OperatorModuleWrapper
        It will return an object of OperatorModuleWrapper, which is the subclass of the original nn.Module.
    """
    if isinstance(func, tvm.ir.module.IRModule) or isinstance(func, tvm.tir.function.PrimFunc):
        return OperatorModuleWrapper(func)
    elif isinstance(func, Callable):
        def func_get_param(*args, **kargs):
            return OperatorModuleWrapper(func(*args, **kargs))
        return func_get_param
