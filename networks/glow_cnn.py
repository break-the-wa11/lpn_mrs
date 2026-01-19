import normflows_1D as nf


class GLOW(nf.MultiscaleFlow):
    def __init__(
        self,
        L,
        K,
        input_shape,
        hidden_channels,
        split_mode = 'channel',
        scale = True,
    ):
        q0_list = []
        merges_list = []
        flows_list = []

        for i in range(L):
            flows_ = []
            for j in range(K):
                flows_ += [
                    nf.flows.GlowBlock(input_shape[0] * 2, hidden_channels, split_mode=split_mode, scale=scale)
                ]
            flows_ += [nf.flows.Squeeze()]
            flows_list += [flows_]

            if i > 0:
                merges_list += [nf.flows.Merge()]
                latent_shape = (input_shape[0], input_shape[1] // 2 ** (L - i))
            else:
                latent_shape = (input_shape[0] * 2, input_shape[1] // 2 ** L)

            q0_list += [nf.distributions.DiagGaussian(latent_shape)]

        super().__init__(q0_list, flows_list, merges_list, class_cond=False)