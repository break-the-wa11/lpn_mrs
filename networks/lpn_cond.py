"""Noise conditioned LPN for 1D signal of shape (1, 512)"""

import numpy as np
import torch
from torch import nn

class LPN_cond(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_c,
        hidden,
        kernel,
        beta,
        alpha,
    ):
        # I set hidden_c to be the channel dim for u (noise conditional parts)
        # I set hidden to be the channel dim for x (signal parts)
        
        super().__init__()

        self.hidden_c = hidden_c
        self.hidden = hidden
        self.kernel = kernel
        self.padding = (kernel - 1) // 2

        self.weight_tilde = nn.ModuleList(
            [
                nn.Linear(1, hidden_c * 512),    # feedforward to transform input into (b, hidden_c * 512) then resize to (b, hidden_c, 512) as sigma
                nn.Conv1d(hidden_c, hidden_c, self.kernel, bias=True, stride=1, padding=self.padding),# 512 u_1
                nn.Conv1d(hidden_c, hidden_c, self.kernel, bias=True, stride=2, padding=self.padding),  # 256 u_2
                nn.Conv1d(hidden_c, hidden_c, self.kernel, bias=True, stride=1, padding=self.padding),  # 256 u_3
                nn.Conv1d(hidden_c, hidden_c, self.kernel, bias=True, stride=2, padding=self.padding),  # 128 u_4
                nn.Conv1d(hidden_c, hidden_c, self.kernel, bias=True, stride=1, padding=self.padding),  # 128 u_5
                nn.Conv1d(hidden_c, hidden_c, self.kernel, bias=True, stride=2, padding=self.padding),  # 64 u_6
            ]
        )

        self.weight_zu = nn.ModuleList(
            [
                nn.Conv1d(hidden_c, hidden, self.kernel, bias=True, stride=1, padding=self.padding),  # 512 match z_1
                nn.Conv1d(hidden_c, hidden, self.kernel, bias=True, stride=1, padding=self.padding),  # 256 match z_2
                nn.Conv1d(hidden_c, hidden, self.kernel, bias=True, stride=1, padding=self.padding),  # 256 match z_3
                nn.Conv1d(hidden_c, hidden, self.kernel, bias=True, stride=1, padding=self.padding),  # 128 match z_4
                nn.Conv1d(hidden_c, hidden, self.kernel, bias=True, stride=1, padding=self.padding),  # 128 match z_5
                nn.Conv1d(hidden_c, hidden, self.kernel, bias=True, stride=1, padding=self.padding),  # 64 match z_6
            ]
        )

        self.weight_z = nn.ModuleList(
            [
                nn.Conv1d(hidden, hidden, self.kernel, bias=False, stride=2, padding=self.padding),  # 256 match z_2
                nn.Conv1d(hidden, hidden, self.kernel, bias=False, stride=1, padding=self.padding),  # 256 match z_3
                nn.Conv1d(hidden, hidden, self.kernel, bias=False, stride=2, padding=self.padding),  # 128 match z_4
                nn.Conv1d(hidden, hidden, self.kernel, bias=False, stride=1, padding=self.padding),  # 128 match z_5
                nn.Conv1d(hidden, hidden, self.kernel, bias=False, stride=2, padding=self.padding),  # 64 match z_6
                nn.Conv1d(hidden, 64, 64, bias=False, stride=1, padding=0),  # (b, 64, 1) z_7
            ]
        )

        self.weight_yu = nn.ModuleList(
            [
                nn.Conv1d(hidden_c, in_dim, self.kernel, bias=True, stride=1, padding=self.padding),  # 512 match signal_size
                nn.Conv1d(hidden_c, in_dim, self.kernel, bias=True, stride=1, padding=self.padding),  # 512 match signal_size
                nn.Conv1d(hidden_c, in_dim, self.kernel, bias=True, stride=1, padding=self.padding),  # 256 match signal_size//2
                nn.Conv1d(hidden_c, in_dim, self.kernel, bias=True, stride=1, padding=self.padding),  # 256 match signal_size//2
                nn.Conv1d(hidden_c, in_dim, self.kernel, bias=True, stride=1, padding=self.padding),  # 128 match signal_size//4
                nn.Conv1d(hidden_c, in_dim, self.kernel, bias=True, stride=1, padding=self.padding),  # 128 match signal_size//4
                nn.Conv1d(hidden_c, in_dim, self.kernel, bias=True, stride=1, padding=self.padding),  # 64 match signal_size//8
            ]
        )

        self.weight_y = nn.ModuleList(
            [
                nn.Conv1d(in_dim, hidden, self.kernel, bias=False, stride=1, padding=self.padding),  # 512 match z_1
                nn.Conv1d(in_dim, hidden, self.kernel, bias=False, stride=2, padding=self.padding),  # 256 match z_2
                nn.Conv1d(in_dim, hidden, self.kernel, bias=False, stride=1, padding=self.padding),  # 256 match z_3
                nn.Conv1d(in_dim, hidden, self.kernel, bias=False, stride=2, padding=self.padding),  # 128 match z_4
                nn.Conv1d(in_dim, hidden, self.kernel, bias=False, stride=1, padding=self.padding),  # 128 match z_5
                nn.Conv1d(in_dim, hidden, self.kernel, bias=False, stride=2, padding=self.padding),  # 64 match z_6
                nn.Conv1d(in_dim, 64, 64, bias=False, stride=1, padding=0),  # (b, 64, 1) z_7
            ]
        )

        self.weight_u = nn.ModuleList(
            [
                nn.Conv1d(hidden_c, hidden, self.kernel, bias=True, stride=1, padding=self.padding),   # 512 match z_1
                nn.Conv1d(hidden_c, hidden, self.kernel, bias=True, stride=2, padding=self.padding),   # 256 match z_2
                nn.Conv1d(hidden_c, hidden, self.kernel, bias=True, stride=1, padding=self.padding),   # 256 match z_3
                nn.Conv1d(hidden_c, hidden, self.kernel, bias=True, stride=2, padding=self.padding),   # 128 match z_4
                nn.Conv1d(hidden_c, hidden, self.kernel, bias=True, stride=1, padding=self.padding),   # 128 match z_5
                nn.Conv1d(hidden_c, hidden, self.kernel, bias=True, stride=2, padding=self.padding),   # 64 match z_6
                nn.Conv1d(hidden_c, 64, 64, bias=True, stride=1, padding=0)   # (b, 64, 1) z_7
            ]
        )
        self.final_layer = nn.Linear(64,1)

        self.act = nn.Softplus(beta=beta)
        self.alpha = alpha
    
    def scalar(self, x, sigma):
        # The input x is of shape (b,1,512)
        # The input of sigma is of shape (b,1)
        bsize = x.shape[0]
        signal_size = x.shape[-1]
        u = sigma.clone()
        z = x.clone()

        size = [
            signal_size,        # z1
            signal_size // 2,   # z2
            signal_size // 2,   # z3
            signal_size // 4,   # z4
            signal_size // 4,   # z5
            signal_size // 8,   # z6
        ]

        #obtain u0 from feedforward and reshape
        u = self.weight_tilde[0](sigma).reshape(bsize, self.hidden_c, signal_size) #transform (b,1) to (b,hidden_c, 512)

        #obtain z1
        z = self.weight_y[0](self.weight_yu[0](u) * x)
        z = z + self.weight_u[0](u)
        z = self.act(z)

        #loop through z2 until z7
        for i in range(0, len(size)):
            u = self.weight_tilde[i+1](u)
            u = self.act(u)

            z = self.weight_z[i](nn.ReLU()(self.weight_zu[i](u)) * z)
            x_scaled = nn.functional.interpolate(x, size[i], mode="linear", align_corners=False)
            z = z + self.weight_y[i+1](self.weight_yu[i+1](u) * x_scaled)
            z = z + self.weight_u[i+1](u)
            z = self.act(z)

        z = z.reshape(bsize,64)
        z = self.final_layer(z)

        z = z + self.alpha * x.reshape(x.shape[0], -1).pow(2).sum(1, keepdim=True)

        return z
    
    # Only init weights that need to be non-negative
    def init_weights(self, mean, std):
        print("init weights")
        with torch.no_grad():
            for core in self.weight_z:
                core.weight.data.normal_(mean, std).exp_()
            self.weight_y[0].weight.data.normal_(mean, std).exp_()

    # this clips the weights to be non-negative to preserve convexity
    def wclip(self):
        with torch.no_grad():
            for core in self.weight_z:
                core.weight.data.clamp_(0)
            self.weight_y[0].weight.data.clamp_(0)

    def forward(self, x, sigma):
        with torch.enable_grad():
            if not x.requires_grad:
                x.requires_grad_(True)
            x_ = x
            sigma_ = sigma
            y = self.scalar(x_, sigma_)
            grad = torch.autograd.grad(
                y.sum(), x_, create_graph=True, retain_graph=True
            )[0]

        return grad