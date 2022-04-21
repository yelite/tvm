# pylint: disable=missing-module-docstring
import torch
import tvm
from typing import List
import torch.utils.dlpack

class TVMScriptModule(torch.nn.Module):
    def __init__(self, module : tvm.runtime.Module):
        super().__init__()
        self.engine = None
        
        # TODO : switch to C++ for milestone 1.2
        # self.engine = torch.classes.tvm_dsoop.TVMScriptModule(module, input_shape, output_shape, self.device, target)
        self.runtime_mod = tvm.build(module)
        # params = [module.buffer_map[x] for x in module.params]
        # self.inputs_shape = params[:-1]
        # self.output_shape = params[-1]

    def forward(self, torch_inputs : List[torch.Tensor]) -> torch.Tensor :
        # TODO : switch to C++ for milestone 1.2
        # r"""Call tvm module to forward"""
        # return self.engine.forward(torch_input)

        tensor_inputs = [tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(i)) for i in torch_inputs]

        # TODO : check the input shape
        # assert tensor_numpy.shape == self.input_shape   

        self.runtime_mod(*tensor_inputs)
        torch_output = tensor_inputs[-1]
        torch_output = torch.utils.dlpack.from_dlpack(torch_output.to_dlpack())
        return torch_output


def as_torch( func: tvm.ir.module.IRModule):
    return TVMScriptModule(func)