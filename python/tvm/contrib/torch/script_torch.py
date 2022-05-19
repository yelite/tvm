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
import torch
import tvm
from typing import List, Union, Callable
import torch.utils.dlpack 
from tvm.target import Target
from tvm.meta_schedule import TuneConfig, tune_tir

class TVMScriptModuleWithCxx(torch.nn.Module):
    def __init__(self, module : Union[tvm.ir.module.IRModule, tvm.tir.function.PrimFunc, tvm.contrib.graph_executor.GraphModule]):
        super().__init__()
        libpt_path = tvm.__path__[0] + "/../../build/libpt_tvmdsoop.so"
        torch.classes.load_library(libpt_path)
        self.engine_cpu = None
        self.engine_cuda = None
        print(type(module))
        if isinstance(module, (tvm.ir.module.IRModule, tvm.tir.function.PrimFunc)):
            self.ir_module = module
            
        elif isinstance(module, tvm.runtime.Module):
            print("here")
            self.__set_relay_module(module)
            
    def __set_relay_module(self, runtime_module):
        func = tvm.get_global_func("tvmtorch.save_runtime_mod")
        func(runtime_module)

        self.engine_cpu = torch.classes.tvm_tuning.RelayRuntime()
        
    def __save_cpu_rt_module(self, runtime_module):
        func = tvm.get_global_func("tvmtorch.save_runtime_mod")
        func(runtime_module)

        self.engine_cpu = torch.classes.tvm_torch.TVMScriptRuntime()
        
    def __build_cpu(self):
        # sch = tvm.tir.Schedule(self.ir_module)
        runtime_module = tvm.build(self.ir_module)
        self.__save_cpu_rt_module(runtime_module)
        
    def __save_cuda_rt_module(self, runtime_module):
        func = tvm.get_global_func("tvmtorch.save_runtime_mod")
        func(runtime_module)

        self.engine_cuda = torch.classes.tvm_torch.TVMScriptRuntime()
        
    def __build_cuda(self):
        # sch = tvm.tir.Schedule(self.ir_module)
        runtime_module = tvm.build(self.ir_module, target=tvm.target.cuda())
        self.__save_cuda_rt_module(runtime_module)
        

    def forward(self, *torch_inputs : List[torch.Tensor]) -> List[torch.Tensor] :
        if torch_inputs[0].is_cuda:
            if self.engine_cuda is None:
                self.__build_cuda()
            return self.engine_cuda.forward(torch_inputs)
        else: 
            if self.engine_cpu is None:
                self.__build_cpu()
            return self.engine_cpu.forward(torch_inputs)
        

# class TVMScriptModule(torch.nn.Module):
#     def __init__(self, module : Union[tvm.ir.module.IRModule, tvm.tir.function.PrimFunc]):
#         super().__init__()
#         self.runtime_mod = tvm.build(module)


#     def forward(self, *torch_inputs : List[torch.Tensor]) -> torch.Tensor :
#         tensor_inputs = [tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(i)) for i in torch_inputs]

#         self.runtime_mod(*tensor_inputs)
#         torch_output = tensor_inputs[-1]
#         torch_output = torch.utils.dlpack.from_dlpack(torch_output.to_dlpack())
#         return torch_output

def as_torch(func: tvm.ir.module.IRModule):
    if isinstance(func, tvm.ir.module.IRModule) or isinstance(func, tvm.tir.function.PrimFunc):
        return TVMScriptModuleWithCxx(func)
    elif isinstance(func, Callable):
        def func_get_param(*args, **kargs):
            return TVMScriptModuleWithCxx(func(*args, **kargs))
        return func_get_param
