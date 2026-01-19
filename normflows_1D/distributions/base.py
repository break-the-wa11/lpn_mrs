import torch
import torch.nn as nn
import numpy as np

from normflows_1D import flows


class BaseDistribution(nn.Module):
    """
    Base distribution of a flow-based model
    Parameters do not depend on target variable
    """

    def __init__(self):
        super().__init__()

    def forward(self, num_samples=1):
        """Samples from base distribution and calculates log probability

        Args:
          num_samples: Number of samples to draw from the distriubtion

        Returns:
          Samples drawn from the distribution, log probability
        """
        raise NotImplementedError

    def log_prob(self, z):
        """Calculate log probability of batch of samples

        Args:
          z: Batch of random variables to determine log probability for

        Returns:
          log probability for each batch element
        """
        raise NotImplementedError

    def sample(self, num_samples=1, **kwargs):
        """Samples from base distribution

        Args:
          num_samples: Number of samples to draw from the distriubtion

        Returns:
          Samples drawn from the distribution
        """
        z, _ = self.forward(num_samples, **kwargs)
        return z
    

class DiagGaussian(BaseDistribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix
    """

    def __init__(self, shape, trainable=True):
        """Constructor

        Args:
          shape: Tuple with shape of data, if int shape has one dimension
          trainable: Flag whether to use trainable or fixed parameters
        """
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)
        if trainable:
            self.loc = nn.Parameter(torch.zeros(1, *self.shape))
            self.log_scale = nn.Parameter(torch.zeros(1, *self.shape))
        else:
            self.register_buffer("loc", torch.zeros(1, *self.shape))
            self.register_buffer("log_scale", torch.zeros(1, *self.shape))
        self.temperature = None  # Temperature parameter for annealed sampling

    def forward(self, num_samples=1, context=None):
        eps = torch.randn(
            (num_samples,) + self.shape, dtype=self.loc.dtype, device=self.loc.device
        )
        if self.temperature is None:
            log_scale = self.log_scale
        else:
            log_scale = self.log_scale + np.log(self.temperature)
        z = self.loc + torch.exp(log_scale) * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow(eps, 2), list(range(1, self.n_dim + 1))
        )
        return z, log_p

    def log_prob(self, z, context=None):
        if self.temperature is None:
            log_scale = self.log_scale
        else:
            log_scale = self.log_scale + np.log(self.temperature)
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow((z - self.loc) / torch.exp(log_scale), 2),
            list(range(1, self.n_dim + 1)),
        )
        return log_p