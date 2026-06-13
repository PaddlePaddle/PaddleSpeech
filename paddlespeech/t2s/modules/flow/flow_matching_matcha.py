from abc import ABC

import paddle
from matcha.models.components.decoder import Decoder
from matcha.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class BASECFM(paddle.nn.Layer, ABC):
    def __init__(self, n_feats, cfm_params, n_spks=1, spk_emb_dim=128):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        if hasattr(cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 0.0001
        self.estimator = None

    @paddle.no_grad()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = paddle.randn(shape=mu.shape, dtype=mu.dtype) * temperature
        t_span = paddle.linspace(start=0, stop=1, num=n_timesteps + 1)
        return self.solve_euler(
            z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond
        )

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        sol = []
        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, mu, t, spks, cond)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return sol[-1]

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape
        t = paddle.rand(shape=[b, 1, 1], dtype=mu.dtype)
        z = paddle.randn(shape=x1.shape, dtype=x1.dtype)
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z
        loss = paddle.nn.functional.mse_loss(
            input=self.estimator(y, mask, mu, t.squeeze(), spks),
            label=u,
            reduction="sum",
        ) / (paddle.sum(mask) * u.shape[1])
        return loss, y


class CFM(BASECFM):
    def __init__(
        self,
        in_channels,
        out_channel,
        cfm_params,
        decoder_params,
        n_spks=1,
        spk_emb_dim=64,
    ):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        in_channels = in_channels + (spk_emb_dim if n_spks > 1 else 0)
        self.estimator = Decoder(
            in_channels=in_channels, out_channels=out_channel, **decoder_params
        )
