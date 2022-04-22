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
    def __init__(self, module : Union[tvm.ir.module.IRModule, tvm.tir.function.PrimFunc]):
        super().__init__()
        # we assume only 1 output for now
        self.engine = torch.classes.tvm_dsoop.TVMScriptModule(len(module.params) - 1, 1, "cpu")
        self.engine.load_IR_module(module)
        

    def forward(self, torch_inputs : List[torch.Tensor]) -> torch.Tensor :
        # tensor_inputs = [tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(i)) for i in torch_inputs]
        # self.runtime_mod(*tensor_inputs)
        # torch_output = tensor_inputs[-1]
        # torch_output = torch.utils.dlpack.from_dlpack(torch_output.to_dlpack())
        # return torch_output
        pass


def as_torch( func: tvm.ir.module.IRModule):
    return TVMScriptModule(func)