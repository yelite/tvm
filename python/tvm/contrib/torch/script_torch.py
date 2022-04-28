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
import torch
import tvm
from typing import List, Union
import torch.utils.dlpack

class TVMScriptModule(torch.nn.Module):
    def __init__(self, module : Union[tvm.ir.module.IRModule, tvm.tir.function.PrimFunc]):
        super().__init__()
        self.runtime_mod = tvm.build(module)


    def forward(self, torch_inputs : List[torch.Tensor]) -> torch.Tensor :
        tensor_inputs = [tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(i)) for i in torch_inputs]

        self.runtime_mod(*tensor_inputs)
        torch_output = tensor_inputs[-1]
        torch_output = torch.utils.dlpack.from_dlpack(torch_output.to_dlpack())
        return torch_output

class TVMScriptModuleWithCxx(torch.nn.Module):
    def __init__(self, ir_module : Union[tvm.ir.module.IRModule, tvm.tir.function.PrimFunc], device):
        super().__init__()
        libpt_path = tvm.__path__[0] + "/../../build/libpt_tvmdsoop.so"
        torch.classes.load_library(libpt_path)

        runtime_module = tvm.build(ir_module, target = "llvm")
        
        func = tvm.get_global_func("tvmtorch.save_runtime_mod")
        func(runtime_module)

        self.engine = torch.classes.tvm_torch.TVMScriptRuntime()
        

    def forward(self, torch_inputs : List[torch.Tensor]) -> None :
        # tensor_inputs = [tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(i)) for i in torch_inputs]
        self.engine.forward(torch_inputs)
        # torch_output = torch.utils.dlpack.from_dlpack(tvm_output)
        


def as_torch(device : str):
    def inner(func: tvm.ir.module.IRModule):
        return TVMScriptModuleWithCxx(func, device)
    return inner