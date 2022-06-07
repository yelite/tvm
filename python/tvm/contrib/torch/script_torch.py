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
from tvm._ffi.runtime_ctypes import Device
from tvm.meta_schedule import TuneConfig
import functools
from tvm import relay
import tempfile
from tvm.meta_schedule.tune import tune_relay
from pathlib import Path
from tvm.runtime.module import load_module
from tvm._ffi.libinfo import find_lib_path


class TVMScriptRtModule(torch.nn.Module):
    def __init__(
        self,
        module: tvm.runtime.Module,
        device: Union[str, Device]
    ):
        super().__init__()
        libpt_path = find_lib_path("libpt_tvmdsoop.so")[0]
        torch.classes.load_library(libpt_path)
        self.__set_relay_module(module)
        self.__device = device
        # self.__rt_module = module

    def __set_relay_module(self, runtime_module):
        func = tvm.get_global_func("tvmtorch.save_runtime_mod")
        func(runtime_module)

        self.engine = torch.classes.tvm_tuning.RelayRuntime()

    def forward(self, *torch_inputs: List[torch.Tensor]):
        ret = self.engine.forward(torch_inputs)
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def save(self, file_name, fmt=""):
        if self.__device.device_type == 1:  # CPU
            self.engine.save_cpu(file_name, fmt)
        elif self.__device.device_type == 2:  # CUDA
            self.engine.save_cuda(file_name, fmt)
        else:
            print(f"Device {self.__device} not supported.")


class TVMScriptIRModule(torch.nn.Module):
    def __init__(self, module: Union[tvm.ir.module.IRModule, tvm.tir.function.PrimFunc, tvm.contrib.graph_executor.GraphModule]):
        super().__init__()
        libpt_path = find_lib_path("libpt_tvmdsoop.so")[0]
        torch.classes.load_library(libpt_path)
        self.engine_cpu = None
        self.engine_cuda = None
        self.ir_module = module

    def __save_cpu_rt_module(self, runtime_module):
        func = tvm.get_global_func("tvmtorch.save_runtime_mod")
        func(runtime_module)

        self.engine_cpu = torch.classes.tvm_torch.TVMScriptRuntime()

    def __build_cpu(self):
        runtime_module = tvm.build(self.ir_module)
        self.__save_cpu_rt_module(runtime_module)

    def __save_cuda_rt_module(self, runtime_module):
        self.engine_cuda = runtime_module

    def __build_cuda(self):
        runtime_module = tvm.build(self.ir_module, target=tvm.target.cuda())
        self.__save_cuda_rt_module(runtime_module)

    def forward(self, *torch_inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        if torch_inputs[0].is_cuda:
            if self.engine_cuda is None:
                self.__build_cuda()
            return self.engine_cuda.forward(torch_inputs)
        else:
            if self.engine_cpu is None:
                self.__build_cpu()
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
    mod : TVMScriptIRModule
        It will return an object of TVMScriptIRModule, which is the subclass of the original nn.Module.
    """
    if isinstance(func, tvm.ir.module.IRModule) or isinstance(func, tvm.tir.function.PrimFunc):
        return TVMScriptIRModule(func)
    elif isinstance(func, Callable):
        def func_get_param(*args, **kargs):
            return TVMScriptIRModule(func(*args, **kargs))
        return func_get_param


@functools.lru_cache(None)
def llvm_target():
    return "llvm -num-cores 16"


def cuda_target():
    return tvm.target.cuda()


def tuning_relay(mod: tvm.ir.module.IRModule, params: Dict, config: TuneConfig, target):
    with tempfile.TemporaryDirectory() as work_dir:
        rt_mod1: tvm.runtime.Module = tune_relay(
            mod=mod,
            params=params,
            target=target,
            config=config,
            work_dir=work_dir,
        )
        return rt_mod1


def optimize_torch(
    func,
    example_inputs,
    tuning_config=None,
    dev=None,
    target=None
):
    """Load PyTorch model that could be traced by TorchScript, then optimize it via MetaSchedule.

    Parameters
    ----------
    func : callable or torch.nn.Module 
        A Python function or nn.Module that could run by TorchScript's trace. (ie: torch.jit.trace(model, input))

    example_inputs : tuple or torch.Tensor 
        A tuple of example inputs that
        will run together with `func` by providing the shape information.

    tuning_config : tvm.meta_schedule.TuneConfig
        The configuration of tuning by MetaSchedule.

    dev : Optional[Union[str, Device]]
        The device to deploy the module. It can be local or remote when there
        is only one Device. 
        If user doesn't set the device, the module is built upon the CPU.

    target : Optional[Union[str, Target]]
        The target of the compilation.
        If user doesn't set the target, the module is built upon the LLVM.

    Returns
    -------
    mod : TVMScriptRtModule
        It will return an object of TVMScriptRtModule, which is the subclass of the original nn.Module.
    """
    if dev:
        pass
    else:
        dev = tvm.cpu(0)
    if target:
        pass
    else:
        target = llvm_target()
    if tuning_config:
        pass
    else:
        tuning_config = TuneConfig(
            strategy="evolutionary",
            num_trials_per_iter=8,
            max_trials_per_task=16,
            max_trials_global=16,
        )
    jit_mod = torch.jit.trace(func, example_inputs)
    if isinstance(example_inputs, torch.Tensor):
        example_inputs = [example_inputs]
    shape_list = [(f"inp_{idx}", i.shape)
                  for idx, i in enumerate(example_inputs)]
    mod, params = relay.frontend.from_pytorch(jit_mod, shape_list)
    mod_after_tuning = tuning_relay(mod, params, tuning_config, target)
    rt_mod = mod_after_tuning["default"](dev)

    return TVMScriptRtModule(rt_mod, dev)


def load_module():
    pass
