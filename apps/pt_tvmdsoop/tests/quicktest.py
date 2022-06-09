import tvm
import torch
from tvm.contrib.torch import optimize_torch
import torch.nn.functional as F

from tvm.meta_schedule.tune import TuneConfig


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5)
        self.conv2 = torch.nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))


tuning_config = TuneConfig(
    strategy="evolutionary",
    num_trials_per_iter=2,
    max_trials_per_task=4,
    max_trials_global=0,
)

rt_mod = optimize_torch(
    SimpleModel(), torch.randn(20, 1, 10, 10), tuning_config)


test_input = torch.randn(20, 1, 10, 10)

ret = rt_mod.forward((test_input,))

torch.save(rt_mod, "test.pt")
loaded_model = torch.load("test.pt")
ret = loaded_model.forward((test_input,))
print(ret)
