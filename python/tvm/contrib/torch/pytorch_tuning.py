import tvm
import torch
import torch.nn.functional as F
from tvm import relay
import functools
from tvm.contrib import graph_executor
import torch.utils.dlpack
from .script_torch import TVMScriptModuleWithCxx
from tvm.meta_schedule import TuneConfig, tune_tir
from tvm.meta_schedule.tune import tune_extracted_tasks, tune_relay
import tempfile
from typing import Union, Dict, List
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

class RelayRTModule(torch.nn.Module):
        def __init__(self, rt_mod : graph_executor.GraphModule):
            super().__init__()
            self.rt_mod = rt_mod

        @staticmethod
        def _to_torch_tensor(nd_tensor : tvm.nd.NDArray) -> torch.Tensor :
            return torch.utils.dlpack.from_dlpack(nd_tensor.to_dlpack())

        def forward(self, *torch_inputs : List[torch.Tensor]) -> torch.Tensor :
            for idx, arg in enumerate(torch_inputs, 0):
                if arg.dim() != 0:
                    self.rt_mod.set_input(
                        f"inp_{idx}",
                        tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(arg)),
                    )
            self.rt_mod.run()
            return [RelayRTModule._to_torch_tensor(self.rt_mod.get_output(i)) for i in range(self.rt_mod.get_num_outputs())]
            
def build_rt_mod(func, example_inputs, tuning_config):
    jit_mod = torch.jit.trace(func, example_inputs)
    shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(example_inputs)]
    mod, params = relay.frontend.from_pytorch(jit_mod, shape_list)
    dev = tvm.cpu(0)
    mod_after_tuning = tuning_relay(mod, params, tuning_config)
    rt_mod = mod_after_tuning["default"](dev)
    
    # rt_mod = graph_executor.GraphModule(rt_mod)
    # return RelayRTModule(rt_mod)
    return TVMScriptModuleWithCxx(rt_mod)
