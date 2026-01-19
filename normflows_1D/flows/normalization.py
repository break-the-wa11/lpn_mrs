import torch

from normflows_1D.flows.affine.coupling import AffineConstFlow


class ActNorm(AffineConstFlow):
    """
    An AffineConstFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """

    def __init__(self, *args, **kwargs):
        """
        Args:
            shape: (tuple) shape of input tensor (not including batch dimension)
            scale: (bool) default true, whether to apply scaling as learnable parameter
            shift: (bool) default true, whether to apply shifting as learnable parameter
        """
        super().__init__(*args, **kwargs)
        self.data_dep_init_done_cpu = torch.tensor(0.0)
        self.register_buffer("data_dep_init_done", self.data_dep_init_done_cpu)

    def forward(self, z):
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done > 0.0:
            assert self.s is not None and self.t is not None
            s_init = -torch.log(z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.s.data = s_init.data
            self.t.data = (
                -z.mean(dim=self.batch_dims, keepdim=True) * torch.exp(self.s)
            ).data
            self.data_dep_init_done = torch.tensor(1.0)
        return super().forward(z)

    def inverse(self, z):
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None
            s_init = torch.log(z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.s.data = s_init.data
            self.t.data = z.mean(dim=self.batch_dims, keepdim=True).data
            self.data_dep_init_done = torch.tensor(1.0)
        return super().inverse(z)