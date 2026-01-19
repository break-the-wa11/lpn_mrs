import torch
from torch import nn

from normflows_1D import flows


class ConstScaleLayer(nn.Module):
    """
    Scaling features by a fixed factor
    """

    def __init__(self, scale=1.0):
        """Constructor

        Args:
          scale: Scale to apply to features
        """
        super().__init__()
        self.scale_cpu = torch.tensor(scale)
        self.register_buffer("scale", self.scale_cpu)

    def forward(self, input):
        return input * self.scale
    

class ActNorm(nn.Module):
    """
    ActNorm layer with just one forward pass and never output log-determinant
    Affine transform of the input
    """
    def __init__(self, shape):
        """Constructor

        Args:
          shape: Same as shape in flows.ActNorm
          logscale_factor: Same as shape in flows.ActNorm

        """
        super().__init__()
        self.actNorm = flows.ActNorm(shape)

    def forward(self, input):
        out, _ = self.actNorm(input)
        return out