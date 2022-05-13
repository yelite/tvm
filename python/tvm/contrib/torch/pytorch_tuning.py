import tvm
import torch
import torch.nn.functional as F
from tvm import relay
import functools
from tvm.contrib import graph_executor
import torch.utils.dlpack
from tvm.meta_schedule import TuneConfig, tune_tir
from tvm.meta_schedule.tune import tune_extracted_tasks, tune_relay
import tempfile
from typing import Union, Dict
from tvm.script import tir as T

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
    rt_mod = graph_executor.GraphModule(mod_after_tuning["default"](dev))
    
    to_torch_tensor = lambda nd_tensor : torch.utils.dlpack.from_dlpack(nd_tensor.to_dlpack())
    
    def exec_tvm(*args):
        args = [a.contiguous() for a in args]
        for idx, arg in enumerate(args, 0):
            if arg.dim() != 0:
                rt_mod.set_input(
                    f"inp_{idx}",
                    tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(arg)),
                )
        rt_mod.run()
        return [to_torch_tensor(rt_mod.get_output(i)) for i in range(rt_mod.get_num_outputs())]

    return exec_tvm
