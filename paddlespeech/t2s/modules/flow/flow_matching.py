import paddle
from abc import ABC
from paddlespeech.t2s.models.CosyVoice.common import set_all_random_seed

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

class ConditionalCFM(BASECFM):
    def __init__(
        self,
        in_channels,
        cfm_params,
        n_spks=1,
        spk_emb_dim=64,
        estimator: paddle.nn.Layer = None,
    ):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        self.estimator = estimator

    @paddle.no_grad()
    def forward(
        self,
        mu,
        mask,
        n_timesteps,
        temperature=1.0,
        spks=None,
        cond=None,
        prompt_len=0,
        cache=paddle.zeros([1, 80, 0, 2]),
    ):
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
        z = (
            paddle.randn(shape=mu.shape, dtype=mu.dtype).to(mu.place).to(mu.dtype)
            * temperature
        )
        cache_size = cache.shape[2]
        if cache_size != 0:
            z[:, :, :cache_size] = cache[:, :, :, 0]
            mu[:, :, :cache_size] = cache[:, :, :, 1]
        z_cache = paddle.cat([z[:, :, :prompt_len], z[:, :, -34:]], axis=2)
        mu_cache = paddle.cat([mu[:, :, :prompt_len], mu[:, :, -34:]], axis=2)
        cache = paddle.stack([z_cache, mu_cache], axis=-1)
        t_span = paddle.linspace(start=0, stop=1, num=n_timesteps + 1, dtype=mu.dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - paddle.cos(t_span * 0.5 * paddle.pi)
        return (
            self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond),
            cache,
        )

    def solve_euler(self, x, t_span, mu, mask, spks, cond, streaming=False):
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
        t = t.unsqueeze(axis=0)
        sol = []
        x_in = paddle.zeros([2, 80, x.shape[2]],  dtype=x.dtype)
        mask_in = paddle.zeros([2, 1, x.shape[2]],  dtype=x.dtype)
        mu_in = paddle.zeros([2, 80, x.shape[2]],  dtype=x.dtype)
        t_in = paddle.zeros([2], dtype=x.dtype)
        spks_in = paddle.zeros([2, 80], dtype=x.dtype)
        cond_in = paddle.zeros([2, 80, x.shape[2]], dtype=x.dtype)
        for step in range(1, len(t_span)):
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t.unsqueeze(0)
            spks_in[0] = spks
            cond_in[0] = cond
            dphi_dt = self.forward_estimator(
                x_in, mask_in, mu_in, t_in, spks_in, cond_in, streaming
            )
            dphi_dt, cfg_dphi_dt = paddle.split(dphi_dt, [x.shape[0], x.shape[0]], axis=0)
            dphi_dt = (
                1.0 + self.inference_cfg_rate
            ) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return sol[-1].float()

    def forward_estimator(self, x, mask, mu, t, spks, cond, streaming=False):
        if isinstance(self.estimator, paddle.nn.Layer):
            return self.estimator(x, mask, mu, t, spks, cond, streaming=streaming)
        else:
            [estimator, stream], trt_engine = self.estimator.acquire_estimator()
            paddle.device.current_stream().synchronize()
            with stream:
                estimator.set_input_shape("x", (2, 80, x.shape[2]))
                estimator.set_input_shape("mask", (2, 1, x.shape[2]))
                estimator.set_input_shape("mu", (2, 80, x.shape[2]))
                estimator.set_input_shape("t", (2,))
                estimator.set_input_shape("spks", (2, 80))
                estimator.set_input_shape("cond", (2, 80, x.shape[2]))
                data_ptrs = [
                    x.contiguous().data_ptr(),
                    mask.contiguous().data_ptr(),
                    mu.contiguous().data_ptr(),
                    t.contiguous().data_ptr(),
                    spks.contiguous().data_ptr(),
                    cond.contiguous().data_ptr(),
                    x.data_ptr(),
                ]
                for i, j in enumerate(data_ptrs):
                    estimator.set_tensor_address(trt_engine.get_tensor_name(i), j)
                assert (
                    estimator.execute_async_v3(
                        paddle.device.current_stream().cuda_stream
                    )
                    is True
                )
                paddle.device.current_stream().synchronize()
            self.estimator.release_estimator(estimator, stream)
            return x

    def compute_loss(self, x1, mask, mu, spks=None, cond=None, streaming=False):
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
        if self.t_scheduler == "cosine":
            t = 1 - paddle.cos(t * 0.5 * paddle.pi)
        z = paddle.randn(shape=x1.shape, dtype=x1.dtype)
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z
        if self.training_cfg_rate > 0:
            cfg_mask = paddle.rand(shape=b) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1, 1)
            spks = spks * cfg_mask.view(-1, 1)
            cond = cond * cfg_mask.view(-1, 1, 1)
        pred = self.estimator(y, mask, mu, t.squeeze(), spks, cond, streaming=streaming)
        loss = paddle.nn.functional.mse_loss(
            input=pred * mask, label=u * mask, reduction="sum"
        ) / (paddle.sum(mask) * u.shape[1])
        return loss, y


class CausalConditionalCFM(ConditionalCFM):
    def __init__(
        self,
        in_channels,
        cfm_params,
        n_spks=1,
        spk_emb_dim=64,
        estimator: paddle.nn.Layer = None,
    ):
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)
        set_all_random_seed(42)
        self.rand_noise = paddle.randn([1, 80, 50 * 300])
    @paddle.no_grad()
    def forward(
        self,
        mu,
        mask,
        n_timesteps,
        temperature=1.0,
        spks=None,
        cond=None,
        streaming=False,
    ):
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
        
        z = self.rand_noise[:, :, : mu.shape[2]].to(mu.place).to(mu.dtype) * temperature
        
        t_span = paddle.linspace(start=0, stop=1, num=n_timesteps + 1, dtype=mu.dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - paddle.cos(t_span * 0.5 * paddle.pi)
        return (
            self.solve_euler(
                z,
                t_span=t_span,
                mu=mu,
                mask=mask,
                spks=spks,
                cond=cond,
                streaming=streaming,
            ),
            None,
        )
