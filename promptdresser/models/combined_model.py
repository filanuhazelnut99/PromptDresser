import torch.nn as nn

class CombinedModel(nn.Module):
    def __init__(self, unet, cloth_encoder):
        super().__init__()
        self.unet = unet
        self.cloth_encoder = cloth_encoder
    def forward(self, x):
        return x

    