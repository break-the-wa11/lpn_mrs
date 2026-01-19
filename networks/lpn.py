"""LPN for 1D signal of shape (1, 512)"""

import numpy as np
import torch
from torch import nn

class LPN(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden,
        kernel,
        beta,
        alpha,
    ):
        super().__init__()

        self.hidden = hidden
        self.kernel = kernel
        self.padding = (kernel - 1) // 2
        self.lin = nn.ModuleList(
            [
                nn.Conv1d(in_dim, hidden, self.kernel, bias=True, stride=1, padding=self.padding),  # 512
                nn.Conv1d(hidden, hidden, self.kernel, bias=False, stride=2, padding=self.padding),  # 256
                nn.Conv1d(hidden, hidden, self.kernel, bias=False, stride=1, padding=self.padding),  # 256
                nn.Conv1d(hidden, hidden, self.kernel, bias=False, stride=2, padding=self.padding),  # 128
                nn.Conv1d(hidden, hidden, self.kernel, bias=False, stride=1, padding=self.padding),  # 128 
                nn.Conv1d(hidden, hidden, self.kernel, bias=False, stride=2, padding=self.padding),  # 64
                nn.Conv1d(hidden, 64, 64, bias=False, stride=1, padding=0),  # 1
                nn.Linear(64, 1),
            ]
        )

        self.res = nn.ModuleList(
            [
                nn.Conv1d(in_dim, hidden, self.kernel, stride=2, padding=self.padding),  # 256
                nn.Conv1d(in_dim, hidden, self.kernel, stride=1, padding=self.padding),  # 256
                nn.Conv1d(in_dim, hidden, self.kernel, stride=2, padding=self.padding),  # 128
                nn.Conv1d(in_dim, hidden, self.kernel, stride=1, padding=self.padding),  # 128
                nn.Conv1d(in_dim, hidden, self.kernel, stride=2, padding=self.padding),  # 64
                nn.Conv1d(in_dim, 64, 64, stride=1, padding=0),  # 1
            ]
        )

        self.act = nn.Softplus(beta=beta)
        self.alpha = alpha
    
    def scalar(self, x):
        bsize = x.shape[0]
        signal_size = x.shape[-1]
        y = x.clone()
        y = self.act(self.lin[0](y))
        size = [
            signal_size,
            signal_size // 2,
            signal_size // 2,
            signal_size // 4,
            signal_size // 4,
            signal_size // 8,
        ]
        for core, res, sz in zip(self.lin[1:-2], self.res[:-1], size[:-1]):
            x_scaled = nn.functional.interpolate(x, sz, mode="linear", align_corners=False)
            y = self.act(core(y) + res(x_scaled))
        
        x_scaled = nn.functional.interpolate(x, size[-1], mode="linear", align_corners=False)
        y = self.lin[-2](y) + self.res[-1](x_scaled)
        y = self.act(y)

        y = y.reshape(bsize, 64)
        y = self.lin[-1](y)     #(batch, 1)

        # strongly convex
        y = y + self.alpha * x.reshape(x.shape[0], -1).pow(2).sum(1, keepdim=True)

        # return shape: (batch, 1)
        return y
    
    def init_weights(self, mean, std):
        print("init weights")
        with torch.no_grad():
            for core in self.lin[1:]:
                core.weight.data.normal_(mean, std).exp_()

    # this clips the weights to be non-negative to preserve convexity
    def wclip(self):
        with torch.no_grad():
            for core in self.lin[1:]:
                core.weight.data.clamp_(0)

    def forward(self, x):
        with torch.enable_grad():
            if not x.requires_grad:
                x.requires_grad_(True)
            x_ = x
            y = self.scalar(x_)
            grad = torch.autograd.grad(
                y.sum(), x_, create_graph=True, retain_graph=True
            )[0]

        return grad