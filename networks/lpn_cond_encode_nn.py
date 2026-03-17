"""Noise conditioned LPN for 1D signal of shape (1, 512)"""

import numpy as np
import torch
from torch import nn

def sigma_encoding(sigma, d_model):
    """
    Generate positional encoding for given sigma.
    
    Args:
        sigma: tensor of scalar values. (b)
        d_model: the dimensionality of the output vector (length of positional encoding).
    
    Returns:
        Tensor of shape (b, d_model) containing the positional encodings.
    """
    sigma = sigma * 1000  # Scale sigma to a larger range for better encoding

    # Create a positional encoding array
    encoding = torch.zeros(sigma.size(0), d_model)

    # Generate the positional encodings
    for pos in range(sigma.size(0)):
        for i in range(0, d_model, 2):
            encoding[pos, i] = torch.sin(sigma[pos] / (10000 ** (i / d_model)))  # Even indices
            if i + 1 < d_model:
                encoding[pos, i + 1] = torch.cos(sigma[pos] / (10000 ** (i / d_model)))  # Odd indices

    return encoding

class LPN_cond_encode_nn(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_c,
        hidden,
        kernel,
        beta,
        alpha,
    ):
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
    
    def _init_weight_tilde(self, s_values=None):
        """
        Initialize self.weight_tilde[0] (nn.Linear(1, hidden_c*512)) so that
        for sample sigmas the linear layer approximates sigma_encoding(sigma).

        s_values: iterable of sigma scalars to fit (if None, choose [0.05,0.1,0.15,0.2])
        """
        linear = self.weight_tilde[0]
        out_dim = linear.out_features  # hidden_c * 512
        in_dim = linear.in_features    # should be 1

        if s_values is None:
            s_values = [0.05, 0.1, 0.15, 0.2]

        # Build design matrix and targets
        S = np.array(s_values, dtype=np.float64).reshape(-1, 1)  # shape (m,1)
        m = S.shape[0]

        # compute target encodings for each sigma using sigma_encoding
        # sigma_encoding expects a torch tensor of shape (m,)
        with torch.no_grad():
            s_tensor = torch.tensor(S.reshape(-1), dtype=torch.float32)
            enc = sigma_encoding(s_tensor, d_model=out_dim)  # returns (m, out_dim) tensor
            T = enc.detach().cpu().numpy()  # shape (m, out_dim)

        # We want linear(s) = W * s + b ≈ T
        # Solve least squares for W (shape out_dim x 1) and b (out_dim,)
        # Augment S with a column of ones for bias
        A = np.concatenate([S, np.ones((m,1), dtype=np.float64)], axis=1)  # (m,2)
        # Solve for each output dimension separately: minimize ||A @ [W_i; b_i] - T[:,i]||^2
        # We can compute solution in one shot using np.linalg.lstsq
        # X shape (2, out_dim) where X = [W; b]^T
        X, *_ = np.linalg.lstsq(A, T, rcond=None)  # returns (2, out_dim)
        # X[0,:] are W row values, X[1,:] are biases

        W = X[0, :].reshape(out_dim, in_dim)  # (out_dim,1)
        b = X[1, :].reshape(out_dim)          # (out_dim,)

        # assign to linear layer (convert to torch float32)
        linear.weight.data = torch.tensor(W, dtype=torch.float32)
        linear.bias.data = torch.tensor(b, dtype=torch.float32)

        # optionally freeze this layer initially if you want fixed embedding
        # for param in linear.parameters():
        #     param.requires_grad = False

    def init_weights(self, mean, std):
        print("init weights")
        with torch.no_grad():
            for core in self.weight_z:
                core.weight.data.normal_(mean, std).exp_()
            for core in self.weight_y:
                core.weight.data.normal_(mean, std).exp_()
            self._init_weight_tilde(s_values=[0.05,0.1,0.15,0.2])

    # this clips the weights to be non-negative to preserve convexity
    def wclip(self):
        with torch.no_grad():
            for core in self.weight_z:
                core.weight.data.clamp_(0)

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