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
from distutils.log import error
import torch
import tvm
from typing import List, Union, Callable
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
    def __init__(self, ir_module : Union[tvm.ir.module.IRModule, tvm.tir.function.PrimFunc], device : str):
        super().__init__()
        libpt_path = tvm.__path__[0] + "/../../build/libpt_tvmdsoop.so"
        torch.classes.load_library(libpt_path)
        
        if device == None:
            runtime_module = tvm.build(ir_module)
        elif device == "cuda":
            raise Exception("Cuda not supported yet")

        func = tvm.get_global_func("tvmtorch.save_runtime_mod")
        func(runtime_module)

        self.engine = torch.classes.tvm_torch.TVMScriptRuntime()
        

    def forward(self, *torch_inputs : List[torch.Tensor]) -> List[torch.Tensor] :
        return self.engine.forward(torch_inputs)
        



def as_torch(device : str = None,):
    def inner(func: tvm.ir.module.IRModule):
        if isinstance(func, tvm.ir.module.IRModule) or isinstance(func, tvm.tir.function.PrimFunc):
            return TVMScriptModuleWithCxx(func, device)
        elif isinstance(func, Callable):
            def func_get_param(*args, **kargs):
                return TVMScriptModuleWithCxx(func(*args, **kargs), device)
            return func_get_param
        
    return inner