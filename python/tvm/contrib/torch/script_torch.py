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
from typing import List, Union, Callable, Dict
import torch.utils.dlpack 
from tvm.meta_schedule import TuneConfig
import functools
from tvm import relay
import tempfile
from tvm.meta_schedule.tune import tune_extracted_tasks, tune_relay

class TVMScriptRtModule(torch.nn.Module):
    def __init__(self, module : tvm.runtime.Module):
        super().__init__()
        libpt_path = tvm.__path__[0] + "/../../build/libpt_tvmdsoop.so"
        torch.classes.load_library(libpt_path)
        self.engine_cpu = None
        self.engine_cuda = None
        self.__set_relay_module(module)
        
    def __set_relay_module(self, runtime_module):
        func = tvm.get_global_func("tvmtorch.save_runtime_mod")
        func(runtime_module)

        self.engine_cpu = torch.classes.tvm_tuning.RelayRuntime()
        
    def forward(self, *torch_inputs : List[torch.Tensor]) -> List[torch.Tensor] :
        if torch_inputs[0].is_cuda:
            if self.engine_cuda is None:
                self.__build_cuda()
            return self.engine_cuda.forward(torch_inputs)
        else: 
            if self.engine_cpu is None:
                self.__build_cpu()
            return self.engine_cpu.forward(torch_inputs)

class TVMScriptIRModule(torch.nn.Module):
    def __init__(self, module : Union[tvm.ir.module.IRModule, tvm.tir.function.PrimFunc, tvm.contrib.graph_executor.GraphModule]):
        super().__init__()
        libpt_path = tvm.__path__[0] + "/../../build/libpt_tvmdsoop.so"
        torch.classes.load_library(libpt_path)
        self.engine_cpu = None
        self.engine_cuda = None
        self.ir_module = module

        
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
        

def as_torch(func: tvm.ir.module.IRModule):
    if isinstance(func, tvm.ir.module.IRModule) or isinstance(func, tvm.tir.function.PrimFunc):
        return TVMScriptIRModule(func)
    elif isinstance(func, Callable):
        def func_get_param(*args, **kargs):
            return TVMScriptIRModule(func(*args, **kargs))
        return func_get_param

@functools.lru_cache(None)
def llvm_target():
    return "llvm -num-cores 16"

def tuning_relay(mod : tvm.ir.module.IRModule, params : Dict, config : TuneConfig):
    with tempfile.TemporaryDirectory() as work_dir:
        rt_mod1: tvm.runtime.Module = tune_relay(
            mod=mod,
            params=params,
            target=llvm_target(),
            config=config,
            work_dir=work_dir,
        )
        return rt_mod1
          
def build_rt_mod(func, example_inputs, tuning_config):
    jit_mod = torch.jit.trace(func, example_inputs)
    shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(example_inputs)]
    mod, params = relay.frontend.from_pytorch(jit_mod, shape_list)
    dev = tvm.cpu(0)
    mod_after_tuning = tuning_relay(mod, params, tuning_config)
    rt_mod = mod_after_tuning["default"](dev)
    
    return TVMScriptRtModule(rt_mod)