# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Diffusion denoising related modules for paddle"""
import math
from typing import Callable
from typing import Optional
from typing import Tuple

import numpy as np
import paddle
import ppdiffusers
from paddle import nn
from ppdiffusers.models.embeddings import Timesteps
from ppdiffusers.schedulers import DDPMScheduler

from paddlespeech.t2s.modules.nets_utils import initialize
from paddlespeech.t2s.modules.residual_block import WaveNetResidualBlock


class GaussianDiffusion(nn.Layer):
    """Common Gaussian Diffusion Denoising Model Module 

    Args:
        denoiser (Layer, optional): 
            The model used for denoising noises.
        num_train_timesteps (int, optional): 
            The number of timesteps between the noise and the real during training, by default 1000.
        beta_start (float, optional): 
            beta start parameter for the scheduler, by default 0.0001.
        beta_end (float, optional): 
            beta end parameter for the scheduler, by default 0.0001.
        beta_schedule (str, optional): 
            beta schedule parameter for the scheduler, by default 'squaredcos_cap_v2' (cosine schedule).
        num_max_timesteps (int, optional): 
            The max timestep transition from real to noise, by default None.
    
    Examples: 
        >>> import paddle
        >>> import paddle.nn.functional as F
        >>> from tqdm import tqdm
        >>> 
        >>> denoiser = WaveNetDenoiser()
        >>> diffusion = GaussianDiffusion(denoiser, num_train_timesteps=1000, num_max_timesteps=100)
        >>> x = paddle.ones([4, 80, 192]) # [B, mel_ch, T] # real mel input
        >>> c = paddle.randn([4, 256, 192]) # [B, fs2_encoder_out_ch, T] # fastspeech2 encoder output
        >>> loss = F.mse_loss(*diffusion(x, c))
        >>> loss.backward()
        >>> print('MSE Loss:', loss.item())
        MSE Loss: 1.6669728755950928 
        >>> def create_progress_callback():
        >>>     pbar = None
        >>>     def callback(index, timestep, num_timesteps, sample):
        >>>         nonlocal pbar
        >>>         if pbar is None:
        >>>             pbar = tqdm(total=num_timesteps)
        >>>             pbar.update(index)
        >>>         pbar.update()
        >>> 
        >>>     return callback
        >>> 
        >>> # ds=1000, K_step=60, scheduler=ddpm, from aux fs2 mel output
        >>> ds = 1000
        >>> infer_steps = 1000
        >>> K_step = 60
        >>> scheduler_type = 'ddpm'
        >>> x_in = x
        >>> diffusion = GaussianDiffusion(denoiser, num_train_timesteps=ds, num_max_timesteps=K_step)
        >>> with paddle.no_grad():
        >>>     sample = diffusion.inference(
        >>>         paddle.randn(x.shape), c, ref_x=x_in, 
        >>>         num_inference_steps=infer_steps,
        >>>         scheduler_type=scheduler_type,
        >>>         callback=create_progress_callback())
        100%|█████| 60/60 [00:03<00:00, 18.36it/s] 
        >>> 
        >>> # ds=100, K_step=100, scheduler=ddpm, from gaussian noise
        >>> ds = 100
        >>> infer_steps = 100
        >>> K_step = 100
        >>> scheduler_type = 'ddpm'
        >>> x_in = None
        >>> diffusion = GaussianDiffusion(denoiser, num_train_timesteps=ds, num_max_timesteps=K_step)
        >>> with paddle.no_grad():
        >>>     sample = diffusion.inference(
        >>>         paddle.randn(x.shape), c, ref_x=x_in, 
        >>>         num_inference_steps=infer_steps,
        >>>         scheduler_type=scheduler_type,
        >>>         callback=create_progress_callback())
        100%|█████| 100/100 [00:05<00:00, 18.29it/s] 
        >>> 
        >>> # ds=1000, K_step=1000, scheduler=pndm, infer_step=25, from gaussian noise
        >>> ds = 1000
        >>> infer_steps = 25
        >>> K_step = 1000
        >>> scheduler_type = 'pndm'
        >>> x_in = None
        >>> diffusion = GaussianDiffusion(denoiser, num_train_timesteps=ds, num_max_timesteps=K_step)
        >>> with paddle.no_grad():
        >>>     sample = diffusion.inference(
        >>>         paddle.randn(x.shape), c, ref_x=x_in, 
        >>>         num_inference_steps=infer_steps,
        >>>         scheduler_type=scheduler_type,
        >>>         callback=create_progress_callback())
        100%|█████| 34/34 [00:01<00:00, 19.75it/s]
        >>> 
        >>> # ds=1000, K_step=100, scheduler=pndm, infer_step=50, from aux fs2 mel output
        >>> ds = 1000
        >>> infer_steps = 50
        >>> K_step = 100
        >>> scheduler_type = 'pndm'
        >>> x_in = x
        >>> diffusion = GaussianDiffusion(denoiser, num_train_timesteps=ds, num_max_timesteps=K_step)
        >>> with paddle.no_grad():
        >>>     sample = diffusion.inference(
        >>>         paddle.randn(x.shape), c, ref_x=x_in, 
        >>>         num_inference_steps=infer_steps,
        >>>         scheduler_type=scheduler_type,
        >>>         callback=create_progress_callback())
        100%|█████| 14/14 [00:00<00:00, 23.80it/s]

    """

    def __init__(
            self,
            denoiser: nn.Layer,
            num_train_timesteps: Optional[int]=1000,
            beta_start: Optional[float]=0.0001,
            beta_end: Optional[float]=0.02,
            beta_schedule: Optional[str]="squaredcos_cap_v2",
            num_max_timesteps: Optional[int]=None,
            stretch: bool=True,
            min_values: paddle.Tensor=None,
            max_values: paddle.Tensor=None, ):
        super().__init__()

        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule

        self.denoiser = denoiser
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule)
        self.num_max_timesteps = num_max_timesteps
        self.stretch = stretch
        self.min_values = min_values
        self.max_values = max_values

    def norm_spec(self, x):
        """
        Linearly map x to [-1, 1]
        Args:
            x: [B, T, N]
        """
        return (x - self.min_values) / (self.max_values - self.min_values
                                        ) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.max_values - self.min_values
                              ) + self.min_values

    def forward(self, x: paddle.Tensor, cond: Optional[paddle.Tensor]=None
                ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Generate random timesteps noised x.

        Args:
            x (Tensor): 
                The input for adding noises.
            cond (Tensor, optional):
                Conditional input for compute noises.
          
        Returns: 
            y (Tensor): 
                The output with noises added in.
            target (Tensor):
                The noises which is added to the input.

        """
        if self.stretch:
            assert self.min_values is not None and self.max_values is not None, "self.min_values and self.max_values should not be None."
            x = x.transpose((0, 2, 1))
            x = self.norm_spec(x)
            x = x.transpose((0, 2, 1))

        noise_scheduler = self.noise_scheduler

        # Sample noise that we'll add to the mel-spectrograms
        target = noise = paddle.randn(x.shape)

        # Sample a random timestep for each mel-spectrogram
        num_timesteps = self.num_train_timesteps
        if self.num_max_timesteps is not None:
            num_timesteps = self.num_max_timesteps
        timesteps = paddle.randint(0, num_timesteps, (x.shape[0], ))

        # Add noise to the clean mel-spectrograms according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = noise_scheduler.add_noise(x, noise, timesteps)

        y = self.denoiser(noisy_images, timesteps, cond)

        # then compute loss use output y and noisy target for prediction_type == "epsilon"
        return y, target

    def inference(self,
                  noise: paddle.Tensor,
                  cond: Optional[paddle.Tensor]=None,
                  ref_x: Optional[paddle.Tensor]=None,
                  num_inference_steps: Optional[int]=1000,
                  strength: Optional[float]=None,
                  scheduler_type: Optional[str]="ddpm",
                  clip_noise: Optional[bool]=True,
                  clip_noise_range: Optional[Tuple[float, float]]=(-1, 1),
                  callback: Optional[Callable[[int, int, int, paddle.Tensor],
                                              None]]=None,
                  callback_steps: Optional[int]=1):
        """Denoising input from noises. Refer to ppdiffusers img2img pipeline.

        Args:
            noise (Tensor): 
                The input tensor as a starting point for denoising. 
            cond (Tensor, optional):
                Conditional input for compute noises. (N, C_aux, T)
            ref_x (Tensor, optional):
                The real output for the denoising process to refer.
            num_inference_steps (int, optional):
                The number of timesteps between the noise and the real during inference, by default 1000.
            strength (float, optional):
                Mixing strength of ref_x with noise. The larger the value, the stronger the noise. 
                Range [0,1], by default None.
            scheduler_type (str, optional):
                Noise scheduler for generate noises. 
                Choose a great scheduler can skip many denoising step, by default 'ddpm'.
                only support 'ddpm' now !
            clip_noise (bool, optional):
                Whether to clip each denoised output, by default True.
            clip_noise_range (tuple, optional):
                denoised output min and max value range after clip, by default (-1, 1).
            callback (Callable[[int,int,int,Tensor], None], optional):
                Callback function during denoising steps.

                Args:
                    index (int):
                        Current denoising index.
                    timestep (int):
                        Current denoising timestep.
                    num_timesteps (int):
                        Number of the denoising timesteps.
                    denoised_output (Tensor):
                        Current intermediate result produced during denoising.

            callback_steps (int, optional):
                The step to call the callback function.
          
        Returns: 
            denoised_output (Tensor): 
                The denoised output tensor.

        """
        scheduler_cls = None
        for clsname in dir(ppdiffusers.schedulers):
            if clsname.lower() == scheduler_type + "scheduler":
                scheduler_cls = getattr(ppdiffusers.schedulers, clsname)
                break

        if scheduler_cls is None:
            raise ValueError(f"No such scheduler type named {scheduler_type}")

        scheduler = scheduler_cls(
            num_train_timesteps=self.num_train_timesteps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule)

        # set timesteps
        scheduler.set_timesteps(num_inference_steps)

        noisy_input = noise
        if self.stretch and ref_x is not None:
            assert self.min_values is not None and self.max_values is not None, "self.min_values and self.max_values should not be None."
            ref_x = ref_x.transpose((0, 2, 1))
            ref_x = self.norm_spec(ref_x)
            ref_x = ref_x.transpose((0, 2, 1))

            # for ddpm
            timesteps = paddle.to_tensor(
                np.flipud(np.arange(num_inference_steps)))
            noisy_input = scheduler.add_noise(ref_x, noise, timesteps[0])

        denoised_output = noisy_input
        for i, t in enumerate(timesteps):
            denoised_output = scheduler.scale_model_input(denoised_output, t)
            noise_pred = self.denoiser(denoised_output, t, cond)
            # compute the previous noisy sample x_t -> x_t-1
            denoised_output = scheduler.step(noise_pred, t,
                                             denoised_output).prev_sample
            if clip_noise:
                denoised_output = paddle.clip(denoised_output, n_min, n_max)

        if self.stretch:
            assert self.min_values is not None and self.max_values is not None, "self.min_values and self.max_values should not be None."
            denoised_output = denoised_output.transpose((0, 2, 1))
            denoised_output = self.denorm_spec(denoised_output)
            denoised_output = denoised_output.transpose((0, 2, 1))

        return denoised_output
