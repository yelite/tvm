import tvm
import torch
from tvm.contrib.torch import optimize_torch, load_module
import torch.nn.functional as F


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5)
        self.conv2 = torch.nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))


simpleModel = optimize_torch(SimpleModel(), torch.randn(20, 1, 10, 10))

simpleModel.save("simple.pt")
