# pylint: disable=missing-module-docstring
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
    def __init__(self, ir_module : Union[tvm.ir.module.IRModule, tvm.tir.function.PrimFunc]):
        super().__init__()
        libpt_path = tvm.__path__[0] + "/../../build/libpt_tvmdsoop.so"
        torch.classes.load_library(libpt_path)

        runtime_module = tvm.build(ir_module)
        
        func = tvm.get_global_func("tvmtorch.save_runtime_mod")
        func(runtime_module)

        self.engine = torch.classes.tvm_torch.TVMScriptRuntime()
        

    def forward(self, torch_inputs : List[torch.Tensor]) -> torch.Tensor :
        print("forward runs")
        tensor_inputs = [tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(i)) for i in torch_inputs]
        tvm_output = self.engine.forward(tensor_inputs)
        torch_output = torch.utils.dlpack.from_dlpack(tvm_output)
        return torch_output
        


def as_torch( func: tvm.ir.module.IRModule):
    return TVMScriptModule(func)