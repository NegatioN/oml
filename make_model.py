import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export, ExportedProgram

'''
Potential models to test: MLP-Mixer
https://github.com/google-research/vision_transformer?tab=readme-ov-file#available-mixer-models
https://medium.com/@tomiwaojo7910/the-mlp-mixer-in-a-nutshell-a-groundbreaking-all-mlp-architecture-for-vision-by-google-researcher-a7f7e614ed01
https://github.com/lucidrains/mlp-mixer-pytorch/tree/main
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 4)
        self.fc2 = nn.Linear(4, 2)
        #self.m = nn.MaxPool1d(3, stride=2)
        # TODO tensor.sum()
        #TODO research hva som er minste nødvendige implementasjon for å få en "saklig" modell til å kjøre

    def forward(self, x):
        x = self.fc1(x) #torch.arange(4, dtype=torch.float32) # 8x1 -> 8x1
        x = F.relu(x)
        x = self.fc2(x)
        print("non-mean", x)
        return x.mean(dim=1)
model = Net()

example_args = (torch.randn(1, 1),)
exported_program: ExportedProgram = export(model, args=example_args)
print(exported_program)
print(model.fc1.weight, model.fc1.bias)
torch.export.save(exported_program, "model.pt2")
print(model.eval()(torch.Tensor([0.5, 1.0, -1.0, 2]).unsqueeze(1)))

'''
## MLP-Mixer
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, dim_hidden, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim_hidden),
        nn.GELU(), # Lets pretend this is a Relu to start off
        nn.Dropout(dropout),
        dense(dim_hidden, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, int(expansion_factor * dim), dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, int(expansion_factor_token * dim), dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )

im_size = 16
model = MLPMixer(
    image_size = im_size,
    channels = 3,
    patch_size = 4,
    dim = 64,
    depth = 1,
    num_classes = 3
)

img = torch.randn(1, 3, im_size, im_size)
example_args = (img,)
exported_program: ExportedProgram = export(model, args=example_args)
print(exported_program)
torch.export.save(exported_program, "model.pt2")
print(model.eval()(img))
'''
