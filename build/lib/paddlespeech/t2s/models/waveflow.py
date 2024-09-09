# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import math
import time
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn import initializer as I

from paddlespeech.t2s.modules import geometry as geo
from paddlespeech.t2s.utils import checkpoint

__all__ = ["WaveFlow", "ConditionalWaveFlow", "WaveFlowLoss"]


def fold(x, n_group):
    """Fold audio or spectrogram's temporal dimension in to groups.

    Args:
        x(Tensor): 
            The input tensor. shape=(*, time_steps)
        n_group(int): 
            The size of a group.

    Returns:
        Tensor: Folded tensor. shape=(*, time_steps // n_group, group)
    """
    spatial_shape = list(x.shape[:-1])
    time_steps = paddle.shape(x)[-1]
    new_shape = spatial_shape + [time_steps // n_group, n_group]
    return paddle.reshape(x, new_shape)


class UpsampleNet(nn.LayerList):
    """Layer to upsample mel spectrogram to the same temporal resolution with
    the corresponding waveform.

    It consists of several conv2dtranspose layers which perform deconvolution
    on mel and time dimension.

    Args:
        upscale_factors(List[int], optional): 
            Time upsampling factors for each Conv2DTranspose Layer.
            The ``UpsampleNet`` contains ``len(upscale_factor)`` Conv2DTranspose
            Layers. Each upscale_factor is used as the ``stride`` for the
            corresponding Conv2DTranspose. Defaults to [16, 16], this the default
            upsampling factor is 256.

    Notes:
        ``np.prod(upscale_factors)`` should equals the ``hop_length`` of the stft
        transformation used to extract spectrogram features from audio.

        For example, ``16 * 16 = 256``, then the spectrogram extracted with a stft
        transformation whose ``hop_length`` equals 256 is suitable.

        See Also
    
        ``librosa.core.stft``
    """

    def __init__(self, upsample_factors):
        super().__init__()
        for factor in upsample_factors:
            std = math.sqrt(1 / (3 * 2 * factor))
            init = I.Uniform(-std, std)
            self.append(
                nn.utils.weight_norm(
                    nn.Conv2DTranspose(
                        1,
                        1, (3, 2 * factor),
                        padding=(1, factor // 2),
                        stride=(1, factor),
                        weight_attr=init,
                        bias_attr=init)))

        # upsample factors
        self.upsample_factor = np.prod(upsample_factors)
        self.upsample_factors = upsample_factors

    def forward(self, x, trim_conv_artifact=False):
        """Forward pass of the ``UpsampleNet``

        Args:
            x(Tensor): 
                The input spectrogram. shape=(batch_size, input_channels, time_steps)
            trim_conv_artifact(bool, optional, optional): 
                Trim deconvolution artifact at each layer. Defaults to False.

        Returns:
           Tensor: The upsampled spectrogram. shape=(batch_size, input_channels, time_steps * upsample_factor)

        Notes:
            If trim_conv_artifact is ``True``, the output time steps is less
            than ``time_steps * upsample_factors``.
        """
        x = paddle.unsqueeze(x, 1)  # (B, C, T) -> (B, 1, C, T)
        for layer in self:
            x = layer(x)
            if trim_conv_artifact:
                time_cutoff = layer._kernel_size[1] - layer._stride[1]
                x = x[:, :, :, :-time_cutoff]
            x = F.leaky_relu(x, 0.4)
        x = paddle.squeeze(x, 1)  # back to (B, C, T)
        return x


class ResidualBlock(nn.Layer):
    """ResidualBlock, the basic unit of ResidualNet used in WaveFlow.

    It has a conv2d layer, which has causal padding in height dimension and
    same paddign in width dimension. It also has projection for the condition
    and output.

    Args:
        channels (int): 
            Feature size of the input.
        cond_channels (int): 
            Featuer size of the condition.
        kernel_size (Tuple[int]): 
            Kernel size of the Convolution2d applied to the input.
        dilations (int): 
            Dilations of the Convolution2d applied to the input.
    """

    def __init__(self, channels, cond_channels, kernel_size, dilations):
        super().__init__()
        # input conv
        std = math.sqrt(1 / channels * np.prod(kernel_size))
        init = I.Uniform(-std, std)
        receptive_field = [
            1 + (k - 1) * d for (k, d) in zip(kernel_size, dilations)
        ]
        rh, rw = receptive_field
        paddings = [rh - 1, 0, rw // 2, (rw - 1) // 2]  # causal & same
        conv = nn.Conv2D(
            channels,
            2 * channels,
            kernel_size,
            padding=paddings,
            dilation=dilations,
            weight_attr=init,
            bias_attr=init)
        self.conv = nn.utils.weight_norm(conv)
        self.rh = rh
        self.rw = rw
        self.dilations = dilations

        # condition projection
        std = math.sqrt(1 / cond_channels)
        init = I.Uniform(-std, std)
        condition_proj = nn.Conv2D(
            cond_channels,
            2 * channels, (1, 1),
            weight_attr=init,
            bias_attr=init)
        self.condition_proj = nn.utils.weight_norm(condition_proj)

        # parametric residual & skip connection
        std = math.sqrt(1 / channels)
        init = I.Uniform(-std, std)
        out_proj = nn.Conv2D(
            channels, 2 * channels, (1, 1), weight_attr=init, bias_attr=init)
        self.out_proj = nn.utils.weight_norm(out_proj)

    def forward(self, x, condition):
        """Compute output for a whole folded sequence.

        Args:
            x (Tensor): 
                The input. [shape=(batch_size, channel, height, width)]
            condition (Tensor [shape=(batch_size, condition_channel, height, width)]): 
                The local condition.

        Returns: 
            res (Tensor): 
                The residual output. [shape=(batch_size, channel, height, width)]
            skip (Tensor): 
                The skip output. [shape=(batch_size, channel, height, width)]
        """
        x_in = x
        x = self.conv(x)
        x += self.condition_proj(condition)

        content, gate = paddle.chunk(x, 2, axis=1)
        x = paddle.tanh(content) * F.sigmoid(gate)

        x = self.out_proj(x)
        res, skip = paddle.chunk(x, 2, axis=1)
        res = x_in + res
        return res, skip

    def start_sequence(self):
        """Prepare the layer for incremental computation of causal
        convolution. Reset the buffer for causal convolution.

        Raises:
            ValueError: If not in evaluation mode.
        """
        if self.training:
            raise ValueError("Only use start sequence at evaluation mode.")
        self._conv_buffer = paddle.zeros([1])

        # NOTE: call self.conv's weight norm hook expliccitly since
        # its weight will be visited directly in `add_input` without
        # calling its `__call__` method. If we do not trigger the weight
        # norm hook, the weight may be outdated. e.g. after loading from
        # a saved checkpoint
        # see also: https://github.com/pytorch/pytorch/issues/47588
        for hook in self.conv._forward_pre_hooks.values():
            hook(self.conv, None)

    def add_input(self, x_row, condition_row):
        """Compute the output for a row and update the buffer.

        Args:
            x_row (Tensor): 
                A row of the input. shape=(batch_size, channel, 1, width)
            condition_row (Tensor): 
                A row of the condition. shape=(batch_size, condition_channel, 1, width)

        Returns:
            res (Tensor): 
                A row of the residual output. shape=(batch_size, channel, 1, width)
            skip (Tensor): 
                A row of the skip output. shape=(batch_size, channel, 1, width)

        """
        x_row_in = x_row
        if len(paddle.shape(self._conv_buffer)) == 1:
            self._init_buffer(x_row)
        self._update_buffer(x_row)
        rw = self.rw
        x_row = F.conv2d(
            self._conv_buffer,
            self.conv.weight,
            self.conv.bias,
            padding=[0, 0, rw // 2, (rw - 1) // 2],
            dilation=self.dilations)
        x_row += self.condition_proj(condition_row)
        content, gate = paddle.chunk(x_row, 2, axis=1)
        x_row = paddle.tanh(content) * F.sigmoid(gate)

        x_row = self.out_proj(x_row)
        res, skip = paddle.chunk(x_row, 2, axis=1)
        res = x_row_in + res
        return res, skip

    def _init_buffer(self, input):
        batch_size, channels, _, width = input.shape
        self._conv_buffer = paddle.zeros(
            [batch_size, channels, self.rh, width], dtype=input.dtype)

    def _update_buffer(self, input):
        self._conv_buffer = paddle.concat(
            [self._conv_buffer[:, :, 1:, :], input], axis=2)


class ResidualNet(nn.LayerList):
    """A stack of several ResidualBlocks. It merges condition at each layer.

    Args:
        n_layer (int): 
            Number of ResidualBlocks in the ResidualNet.
        residual_channels (int): 
            Feature size of each ResidualBlocks.
        condition_channels (int): 
            Feature size of the condition.
        kernel_size (Tuple[int]): 
            Kernel size of each ResidualBlock.
        dilations_h (List[int]): 
            Dilation in height dimension of every ResidualBlock.

    Raises:
        ValueError: If the length of dilations_h does not equals n_layers.
    """

    def __init__(self,
                 n_layer: int,
                 residual_channels: int,
                 condition_channels: int,
                 kernel_size: Tuple[int],
                 dilations_h: List[int]):
        if len(dilations_h) != n_layer:
            raise ValueError(
                "number of dilations_h should equals num of layers")
        super().__init__()
        for i in range(n_layer):
            dilation = (dilations_h[i], 2**i)
            layer = ResidualBlock(residual_channels, condition_channels,
                                  kernel_size, dilation)
            self.append(layer)

    def forward(self, x, condition):
        """Comput the output of given the input and the condition.

        Args:
            x (Tensor): 
                The input. shape=(batch_size, channel, height, width)
            condition (Tensor): 
                The local condition. shape=(batch_size, condition_channel, height, width)
            
        Returns: 
            Tensor: The output, which is an aggregation of all the skip outputs. shape=(batch_size, channel, height, width)
            
        """
        skip_connections = []
        for layer in self:
            x, skip = layer(x, condition)
            skip_connections.append(skip)
        out = paddle.sum(paddle.stack(skip_connections, 0), 0)
        return out

    def start_sequence(self):
        """Prepare the layer for incremental computation.
        """
        for layer in self:
            layer.start_sequence()

    def add_input(self, x_row, condition_row):
        """Compute the output for a row and update the buffers.

        Args:
            x_row (Tensor): 
                A row of the input. shape=(batch_size, channel, 1, width)
            condition_row (Tensor):  
                A row of the condition. shape=(batch_size, condition_channel, 1, width)
            
        Returns:
            res (Tensor): 
                A row of the residual output. shape=(batch_size, channel, 1, width) 
            skip (Tensor): 
                A row of the skip output. shape=(batch_size, channel, 1, width)
                
        """
        skip_connections = []
        for layer in self:
            x_row, skip = layer.add_input(x_row, condition_row)
            skip_connections.append(skip)
        out = paddle.sum(paddle.stack(skip_connections, 0), 0)
        return out


class Flow(nn.Layer):
    """A bijection (Reversable layer) that transform a density of latent
    variables p(Z) into a complex data distribution p(X).

    It's an auto regressive flow. The ``forward`` method implements the
    probability density estimation. The ``inverse`` method implements the
    sampling.

    Args:
        n_layers (int): 
            Number of ResidualBlocks in the Flow.
        channels (int): 
            Feature size of the ResidualBlocks.
        mel_bands (int): 
            Feature size of the mel spectrogram (mel bands).
        kernel_size (Tuple[int]): 
            Kernel size of each ResisualBlocks in the Flow.
        n_group (int): 
            Number of timesteps to the folded into a group.
    """
    dilations_dict = {
        8: [1, 1, 1, 1, 1, 1, 1, 1],
        16: [1, 1, 1, 1, 1, 1, 1, 1],
        32: [1, 2, 4, 1, 2, 4, 1, 2],
        64: [1, 2, 4, 8, 16, 1, 2, 4],
        128: [1, 2, 4, 8, 16, 32, 64, 1]
    }

    def __init__(self, n_layers, channels, mel_bands, kernel_size, n_group):
        super().__init__()
        # input projection
        self.input_proj = nn.utils.weight_norm(
            nn.Conv2D(
                1,
                channels, (1, 1),
                weight_attr=I.Uniform(-1., 1.),
                bias_attr=I.Uniform(-1., 1.)))

        # residual net
        self.resnet = ResidualNet(n_layers, channels, mel_bands, kernel_size,
                                  self.dilations_dict[n_group])

        # output projection
        self.output_proj = nn.Conv2D(
            channels,
            2, (1, 1),
            weight_attr=I.Constant(0.),
            bias_attr=I.Constant(0.))

        # specs
        self.n_group = n_group

    def _predict_parameters(self, x, condition):
        x = self.input_proj(x)
        x = self.resnet(x, condition)
        bijection_params = self.output_proj(x)
        logs, b = paddle.chunk(bijection_params, 2, axis=1)
        return logs, b

    def _transform(self, x, logs, b):
        z_0 = x[:, :, :1, :]  # the first row, just copy it
        z_out = x[:, :, 1:, :] * paddle.exp(logs) + b
        z_out = paddle.concat([z_0, z_out], axis=2)
        return z_out

    def forward(self, x, condition):
        """Probability density estimation. It is done by inversely transform
        a sample from p(X) into a sample from p(Z).

        Args:
            x (Tensor): 
                A input sample of the distribution p(X). shape=(batch, 1, height, width)
            condition (Tensor): 
                The local condition. shape=(batch, condition_channel, height, width)
            
        Returns:
            z (Tensor): 
                shape(batch, 1, height, width), the transformed sample.
            Tuple[Tensor, Tensor]:
                The parameter of the transformation.
                logs (Tensor): shape(batch, 1, height - 1, width), the log scale of the transformation from x to z.
                b (Tensor): shape(batch, 1, height - 1, width), the shift of the transformation from x to z.
        """
        # (B, C, H-1, W)
        logs, b = self._predict_parameters(x[:, :, :-1, :],
                                           condition[:, :, 1:, :])
        z = self._transform(x, logs, b)
        return z, (logs, b)

    def _predict_row_parameters(self, x_row, condition_row):
        x_row = self.input_proj(x_row)
        x_row = self.resnet.add_input(x_row, condition_row)
        bijection_params = self.output_proj(x_row)
        logs, b = paddle.chunk(bijection_params, 2, axis=1)
        return logs, b

    def _inverse_transform_row(self, z_row, logs, b):
        x_row = (z_row - b) * paddle.exp(-logs)
        return x_row

    def _inverse_row(self, z_row, x_row, condition_row):
        logs, b = self._predict_row_parameters(x_row, condition_row)
        x_next_row = self._inverse_transform_row(z_row, logs, b)
        return x_next_row, (logs, b)

    def _start_sequence(self):
        self.resnet.start_sequence()

    def inverse(self, z, condition):
        """Sampling from the distrition p(X). It is done by sample form
        p(Z) and transform the sample. It is a auto regressive transformation.

        Args:
            z(Tensor): 
                A sample of the distribution p(Z). shape=(batch, 1, time_steps
            condition(Tensor): 
                The local condition. shape=(batch, condition_channel, time_steps)
        Returns:
            Tensor:
                The transformed sample. shape=(batch, 1, height, width)
        """
        z_0 = z[:, :, :1, :]
        x = paddle.zeros_like(z)
        x[:, :, :1, :] = z_0

        self._start_sequence()

        num_step = paddle.ones([1], dtype='int32') * (self.n_group)
        for i in range(1, num_step):
            x_row = x[:, :, i - 1:i, :]
            z_row = z[:, :, i:i + 1, :]
            condition_row = condition[:, :, i:i + 1, :]
            x_next_row, (logs, b) = self._inverse_row(z_row, x_row,
                                                      condition_row)
            x[:, :, i:i + 1, :] = x_next_row

        return x


class WaveFlow(nn.LayerList):
    """An Deep Reversible layer that is composed of severel auto regressive
    flows.

    Args:
        n_flows (int): 
            Number of flows in the WaveFlow model.
        n_layers (int): 
            Number of ResidualBlocks in each Flow.
        n_group (int): 
            Number of timesteps to fold as a group.
        channels (int): 
            Feature size of each ResidualBlock.
        mel_bands (int): 
            Feature size of mel spectrogram (mel bands).
        kernel_size (Union[int, List[int]]): 
            Kernel size of the convolution layer in each ResidualBlock.
    """

    def __init__(self, n_flows, n_layers, n_group, channels, mel_bands,
                 kernel_size):
        if n_group % 2 or n_flows % 2:
            raise ValueError(
                "number of flows and number of group must be even "
                "since a permutation along group among flows is used.")
        super().__init__()
        for _ in range(n_flows):
            self.append(
                Flow(n_layers, channels, mel_bands, kernel_size, n_group))

        # permutations in h
        self.perms = self._create_perm(n_group, n_flows)

        # specs
        self.n_group = n_group
        self.n_flows = n_flows

    def _create_perm(self, n_group, n_flows):
        indices = list(range(n_group))
        half = n_group // 2
        perms = []
        for i in range(n_flows):
            if i < n_flows // 2:
                perm = indices[::-1]
            else:
                perm = list(reversed(indices[:half])) + list(
                    reversed(indices[half:]))
            perm = paddle.to_tensor(perm)
            self.register_buffer(perm.name, perm)
            perms.append(perm)
        return perms

    def _trim(self, x, condition):
        assert condition.shape[-1] >= x.shape[-1]
        pruned_len = int(paddle.shape(x)[-1] // self.n_group * self.n_group)

        if x.shape[-1] > pruned_len:
            x = x[:, :pruned_len]
        if condition.shape[-1] > pruned_len:
            condition = condition[:, :, :pruned_len]
        return x, condition

    def forward(self, x, condition):
        """Probability density estimation of random variable x given the
        condition.

        Args:
            x (Tensor): 
                The audio. shape=(batch_size, time_steps)
            condition (Tensor): 
                The local condition (mel spectrogram here). shape=(batch_size, condition channel, time_steps)
                
        Returns:
            Tensor: 
                The transformed random variable. shape=(batch_size, time_steps)
            Tensor: 
                The log determinant of the jacobian of the transformation from x to z. shape=(1,)
        """
        # x: (B, T)
        # condition: (B, C, T) upsampled condition
        x, condition = self._trim(x, condition)

        # to (B, C, h, T//h) layout
        x = paddle.unsqueeze(
            paddle.transpose(fold(x, self.n_group), [0, 2, 1]), 1)
        condition = paddle.transpose(
            fold(condition, self.n_group), [0, 1, 3, 2])

        # flows
        logs_list = []
        for i, layer in enumerate(self):
            x, (logs, b) = layer(x, condition)
            logs_list.append(logs)
            # permute paddle has no shuffle dim
            x = geo.shuffle_dim(x, 2, perm=self.perms[i])
            condition = geo.shuffle_dim(condition, 2, perm=self.perms[i])

        z = paddle.squeeze(x, 1)  # (B, H, W)
        batch_size = z.shape[0]
        z = paddle.reshape(paddle.transpose(z, [0, 2, 1]), [batch_size, -1])

        log_det_jacobian = paddle.sum(paddle.stack(logs_list))
        return z, log_det_jacobian

    def inverse(self, z, condition):
        """Sampling from the distrition p(X).

        It is done by sample a ``z`` form p(Z) and transform it into ``x``.
        Each Flow transform .. math:: `z_{i-1}` to .. math:: `z_{i}` in an
        autoregressive manner.

        Args:
            z (Tensor): 
                A sample of the distribution p(Z). shape=(batch, 1, time_steps
            condition (Tensor): 
                The local condition. shape=(batch, condition_channel, time_steps)    

        Returns: 
            Tensor: The transformed sample (audio here). shape=(batch_size, time_steps)
        """

        z, condition = self._trim(z, condition)
        # to (B, C, h, T//h) layout
        z = paddle.unsqueeze(
            paddle.transpose(fold(z, self.n_group), [0, 2, 1]), 1)
        condition = paddle.transpose(
            fold(condition, self.n_group), [0, 1, 3, 2])

        # reverse it flow by flow
        for i in reversed(range(self.n_flows)):
            z = geo.shuffle_dim(z, 2, perm=self.perms[i])
            condition = geo.shuffle_dim(condition, 2, perm=self.perms[i])
            z = self[i].inverse(z, condition)

        x = paddle.squeeze(z, 1)  # (B, H, W)
        batch_size = x.shape[0]
        x = paddle.reshape(paddle.transpose(x, [0, 2, 1]), [batch_size, -1])
        return x


class ConditionalWaveFlow(nn.LayerList):
    """ConditionalWaveFlow, a UpsampleNet with a WaveFlow model.

    Args:
        upsample_factors (List[int]): 
            Upsample factors for the upsample net.
        n_flows (int): 
            Number of flows in the WaveFlow model.
        n_layers (int): 
            Number of ResidualBlocks in each Flow.
        n_group (int): 
            Number of timesteps to fold as a group.
        channels (int): 
            Feature size of each ResidualBlock.
        n_mels (int): 
            Feature size of mel spectrogram (mel bands).
        kernel_size (Union[int, List[int]]): 
            Kernel size of the convolution layer in each ResidualBlock.
        """

    def __init__(self,
                 upsample_factors: List[int],
                 n_flows: int,
                 n_layers: int,
                 n_group: int,
                 channels: int,
                 n_mels: int,
                 kernel_size: Union[int, List[int]]):
        super().__init__()
        self.encoder = UpsampleNet(upsample_factors)
        self.decoder = WaveFlow(
            n_flows=n_flows,
            n_layers=n_layers,
            n_group=n_group,
            channels=channels,
            mel_bands=n_mels,
            kernel_size=kernel_size)

    def forward(self, audio, mel):
        """Compute the transformed random variable z (x to z) and the log of
        the determinant of the jacobian of the transformation from x to z.

        Args:
            audio(Tensor): 
                The audio. shape=(B, T)
            mel(Tensor): 
                The mel spectrogram. shape=(B, C_mel, T_mel)

        Returns:
            Tensor: 
                The inversely transformed random variable z (x to z). shape=(B, T)
            Tensor: 
                the log of the determinant of the jacobian of the transformation from x to z. shape=(1,)
        """
        condition = self.encoder(mel)
        z, log_det_jacobian = self.decoder(audio, condition)
        return z, log_det_jacobian

    @paddle.no_grad()
    def infer(self, mel):
        """Generate raw audio given mel spectrogram.

        Args:
            mel(np.ndarray): 
                Mel spectrogram of an utterance(in log-magnitude). shape=(C_mel, T_mel)

        Returns:
            Tensor: 
                The synthesized audio, where``T <= T_mel * upsample_factors``. shape=(B, T)
        """
        start = time.time()
        condition = self.encoder(mel, trim_conv_artifact=True)  # (B, C, T)
        batch_size, _, time_steps = condition.shape
        z = paddle.randn([batch_size, time_steps], dtype=mel.dtype)
        x = self.decoder.inverse(z, condition)
        end = time.time()
        print("time: {}s".format(end - start))
        return x

    @paddle.no_grad()
    def predict(self, mel):
        """Generate raw audio given mel spectrogram.

        Args:
            mel(np.ndarray): 
                Mel spectrogram of an utterance(in log-magnitude). shape=(C_mel, T_mel)

        Returns:
            np.ndarray: The synthesized audio. shape=(T,)
        """
        mel = paddle.to_tensor(mel)
        mel = paddle.unsqueeze(mel, 0)
        audio = self.infer(mel)
        audio = audio[0].numpy()
        return audio

    @classmethod
    def from_pretrained(cls, config, checkpoint_path):
        """Build a ConditionalWaveFlow model from a pretrained model.

        Args:
            config(yacs.config.CfgNode): 
                model configs
            checkpoint_path(Path or str): 
                the path of pretrained model checkpoint, without extension name

        Returns:
            ConditionalWaveFlow The model built from pretrained result.
        """
        model = cls(upsample_factors=config.model.upsample_factors,
                    n_flows=config.model.n_flows,
                    n_layers=config.model.n_layers,
                    n_group=config.model.n_group,
                    channels=config.model.channels,
                    n_mels=config.data.n_mels,
                    kernel_size=config.model.kernel_size)
        checkpoint.load_parameters(model, checkpoint_path=checkpoint_path)
        return model


class WaveFlowLoss(nn.Layer):
    """Criterion of a WaveFlow model.

    Args:
        sigma (float): 
            The standard deviation of the gaussian noise used in WaveFlow, by default 1.0.
    """

    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        self.const = 0.5 * np.log(2 * np.pi) + np.log(self.sigma)

    def forward(self, z, log_det_jacobian):
        """Compute the loss given the transformed random variable z and the
        log_det_jacobian of transformation from x to z.

        Args:
            z(Tensor): 
                The transformed random variable (x to z). shape=(B, T)
            log_det_jacobian(Tensor): 
                The log of the determinant of the jacobian matrix of the
                transformation from x to z.  shape=(1,)

        Returns:
            Tensor: The loss. shape=(1,)
        """
        loss = paddle.sum(z * z) / (2 * self.sigma * self.sigma
                                    ) - log_det_jacobian
        loss = loss / np.prod(z.shape)
        return loss + self.const


class ConditionalWaveFlow2Infer(ConditionalWaveFlow):
    def forward(self, mel):
        """Generate raw audio given mel spectrogram.

        Args:
            mel (np.ndarray): 
                Mel spectrogram of an utterance(in log-magnitude). shape=(C_mel, T_mel)
            
        Returns:
            np.ndarray: The synthesized audio. shape=(T,)
            
        """
        audio = self.predict(mel)
        return audio
