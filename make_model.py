import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export, ExportedProgram

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.fc1(x) + torch.arange(4, dtype=torch.float32) # 8x1 -> 8x1
        x = self.fc2(x)  # 8x1 -> 8x1
        return F.relu(x)
model = Net()

example_args = (torch.randn(1, 1),)
exported_program: ExportedProgram = export(model, args=example_args)
print(exported_program)
print(model.fc1.weight, model.fc1.bias)
torch.export.save(exported_program, "model.pt")
print(model.eval()(torch.Tensor([0.5, 1.0, -1.0, 2]).unsqueeze(1)))
#TODO unzip model as well, as it makes it easier for use

