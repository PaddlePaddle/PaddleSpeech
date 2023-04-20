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
"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""
import math

import paddle
import paddle.nn.functional as F
from paddle import nn

from paddlespeech.t2s.modules.nets_utils import _reset_parameters


class DownSample(nn.Layer):
    def __init__(self, layer_type: str):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x: paddle.Tensor):
        """Calculate forward propagation.
        Args:
            x(Tensor(float32)): Shape (B, dim_in, n_mels, T).
        Returns:
            Tensor:
                layer_type == 'none': Shape (B, dim_in, n_mels, T)
                layer_type == 'timepreserve': Shape (B, dim_in, n_mels // 2, T)
                layer_type == 'half': Shape (B, dim_in, n_mels // 2, T // 2)
        """
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            out = F.avg_pool2d(x, (2, 1))
            return out
        elif self.layer_type == 'half':
            out = F.avg_pool2d(x, 2)
            return out
        else:
            raise RuntimeError(
                'Got unexpected donwsampletype %s, expected is [none, timepreserve, half]'
                % self.layer_type)


class UpSample(nn.Layer):
    def __init__(self, layer_type: str):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x: paddle.Tensor):
        """Calculate forward propagation.
        Args:
            x(Tensor(float32)): Shape (B, dim_in, n_mels, T).
        Returns:
            Tensor:
                layer_type == 'none': Shape (B, dim_in, n_mels, T)
                layer_type == 'timepreserve': Shape (B, dim_in, n_mels * 2, T)
                layer_type == 'half': Shape (B, dim_in, n_mels * 2, T * 2)
        """
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            out = F.interpolate(x, scale_factor=(2, 1), mode='nearest')
            return out
        elif self.layer_type == 'half':
            out = F.interpolate(x, scale_factor=2, mode='nearest')
            return out
        else:
            raise RuntimeError(
                'Got unexpected upsampletype %s, expected is [none, timepreserve, half]'
                % self.layer_type)


class ResBlk(nn.Layer):
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 actv: nn.LeakyReLU=nn.LeakyReLU(0.2),
                 normalize: bool=False,
                 downsample: str='none'):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(layer_type=downsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in: int, dim_out: int):
        self.conv1 = nn.Conv2D(
            in_channels=dim_in,
            out_channels=dim_in,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv2 = nn.Conv2D(
            in_channels=dim_in,
            out_channels=dim_out,
            kernel_size=3,
            stride=1,
            padding=1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2D(dim_in)
            self.norm2 = nn.InstanceNorm2D(dim_in)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2D(
                in_channels=dim_in,
                out_channels=dim_out,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False)

    def _shortcut(self, x: paddle.Tensor):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x: paddle.Tensor):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x: paddle.Tensor):
        """Calculate forward propagation.
        Args:
            x(Tensor(float32)): Shape (B, dim_in, n_mels, T).
        Returns:
            Tensor:
                downsample == 'none': Shape (B, dim_in, n_mels, T).
                downsample == 'timepreserve': Shape (B, dim_out, T, n_mels // 2, T).
                downsample == 'half': Shape (B, dim_out, T, n_mels // 2, T // 2).
        """
        x = self._shortcut(x) + self._residual(x)
        # unit variance
        out = x / math.sqrt(2)
        return out


class AdaIN(nn.Layer):
    def __init__(self, style_dim: int, num_features: int):
        super().__init__()
        self.norm = nn.InstanceNorm2D(
            num_features=num_features, weight_attr=False, bias_attr=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x: paddle.Tensor, s: paddle.Tensor):
        """Calculate forward propagation.
        Args:
            x(Tensor(float32)): Shape (B, style_dim, n_mels, T).
            s(Tensor(float32)): Shape (style_dim, ).
        Returns:
            Tensor:
                Shape (B, style_dim, T, n_mels, T).
        """
        if len(s.shape) == 1:
            s = s[None]
        h = self.fc(s)
        h = h.reshape((h.shape[0], h.shape[1], 1, 1))
        gamma, beta = paddle.split(h, 2, axis=1)
        out = (1 + gamma) * self.norm(x) + beta
        return out


class AdainResBlk(nn.Layer):
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 style_dim: int=64,
                 w_hpf: int=0,
                 actv: nn.Layer=nn.LeakyReLU(0.2),
                 upsample: str='none'):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = UpSample(layer_type=upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.layer_type = upsample

    def _build_weights(self, dim_in: int, dim_out: int, style_dim: int=64):
        self.conv1 = nn.Conv2D(
            in_channels=dim_in,
            out_channels=dim_out,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv2 = nn.Conv2D(
            in_channels=dim_out,
            out_channels=dim_out,
            kernel_size=3,
            stride=1,
            padding=1)
        self.norm1 = AdaIN(style_dim=style_dim, num_features=dim_in)
        self.norm2 = AdaIN(style_dim=style_dim, num_features=dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2D(
                in_channels=dim_in,
                out_channels=dim_out,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False)

    def _shortcut(self, x: paddle.Tensor):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x: paddle.Tensor, s: paddle.Tensor):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x: paddle.Tensor, s: paddle.Tensor):
        """Calculate forward propagation.
        Args:
            x(Tensor(float32)): 
                Shape (B, dim_in, n_mels, T).
            s(Tensor(float32)):
                Shape (64,).
        Returns:
            Tensor:
                upsample == 'none': Shape (B, dim_out, T, n_mels, T).  
                upsample == 'timepreserve': Shape (B, dim_out, T, n_mels * 2, T).
                upsample == 'half': Shape (B, dim_out, T, n_mels * 2, T * 2).  
        """
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class HighPass(nn.Layer):
    def __init__(self, w_hpf: int):
        super().__init__()
        self.filter = paddle.to_tensor([[-1, -1, -1], [-1, 8., -1],
                                        [-1, -1, -1]]) / w_hpf

    def forward(self, x: paddle.Tensor):
        filter = self.filter.unsqueeze(0).unsqueeze(1).tile(
            [x.shape[1], 1, 1, 1])
        out = F.conv2d(x, filter, padding=1, groups=x.shape[1])
        return out


class Generator(nn.Layer):
    def __init__(self,
                 dim_in: int=48,
                 style_dim: int=48,
                 max_conv_dim: int=48 * 8,
                 w_hpf: int=1,
                 F0_channel: int=0):
        super().__init__()

        self.stem = nn.Conv2D(
            in_channels=1,
            out_channels=dim_in,
            kernel_size=3,
            stride=1,
            padding=1)
        self.encode = nn.LayerList()
        self.decode = nn.LayerList()
        self.to_out = nn.Sequential(
            nn.InstanceNorm2D(dim_in),
            nn.LeakyReLU(0.2),
            nn.Conv2D(
                in_channels=dim_in,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0))
        self.F0_channel = F0_channel
        # down/up-sampling blocks
        # int(np.log2(img_size)) - 4
        repeat_num = 4
        if w_hpf > 0:
            repeat_num += 1

        for lid in range(repeat_num):
            if lid in [1, 3]:
                _downtype = 'timepreserve'
            else:
                _downtype = 'half'

            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(
                ResBlk(
                    dim_in=dim_in,
                    dim_out=dim_out,
                    normalize=True,
                    downsample=_downtype))
            (self.decode.insert if lid else
             lambda i, sublayer: self.decode.append(sublayer))(0, AdainResBlk(
                 dim_in=dim_out,
                 dim_out=dim_in,
                 style_dim=style_dim,
                 w_hpf=w_hpf,
                 upsample=_downtype))  # stack-like
            dim_in = dim_out
        # bottleneck blocks (encoder)
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_in=dim_out, dim_out=dim_out, normalize=True))
        # F0 blocks 
        if F0_channel != 0:
            self.decode.insert(0,
                               AdainResBlk(
                                   dim_in=dim_out + int(F0_channel / 2),
                                   dim_out=dim_out,
                                   style_dim=style_dim,
                                   w_hpf=w_hpf))
        # bottleneck blocks (decoder)
        for _ in range(2):
            self.decode.insert(0,
                               AdainResBlk(
                                   dim_in=dim_out + int(F0_channel / 2),
                                   dim_out=dim_out + int(F0_channel / 2),
                                   style_dim=style_dim,
                                   w_hpf=w_hpf))
        if F0_channel != 0:
            self.F0_conv = nn.Sequential(
                ResBlk(
                    dim_in=F0_channel,
                    dim_out=int(F0_channel / 2),
                    normalize=True,
                    downsample="half"), )
        if w_hpf > 0:
            self.hpf = HighPass(w_hpf)

        self.reset_parameters()

    def forward(self,
                x: paddle.Tensor,
                s: paddle.Tensor,
                masks: paddle.Tensor=None,
                F0: paddle.Tensor=None):
        """Calculate forward propagation.
        Args:
            x(Tensor(float32)): 
                Shape (B, 1, n_mels, T).
            s(Tensor(float32)):
                Shape (64,).
            masks:
                None.
            F0:
                Shape (B, num_features(256), n_mels // 8, T).
        Returns:
            Tensor:
                output of generator. Shape (B, 1, n_mels, T // 4 * 4)
        """
        x = self.stem(x)
        cache = {}
        # output: (B, max_conv_dim, n_mels // 16, T // 4)
        for block in self.encode:
            if (masks is not None) and (x.shape[2] in [32, 64, 128]):
                cache[x.shape[2]] = x
            x = block(x)
        if F0 is not None:
            # input: (B, num_features(256), n_mels // 8, T)
            # output: (B, num_features(256) // 2, n_mels // 16, T // 2)
            F0 = self.F0_conv(F0)
            # output: (B, num_features(256) // 2, n_mels // 16, T // 4)
            F0 = F.adaptive_avg_pool2d(F0, [x.shape[-2], x.shape[-1]])
            x = paddle.concat([x, F0], axis=1)
        # input: (B, max_conv_dim+num_features(256) // 2, n_mels // 16, T // 4 * 4)
        # output: (B, dim_in, n_mels, T // 4 * 4)
        for block in self.decode:
            x = block(x, s)
            if (masks is not None) and (x.shape[2] in [32, 64, 128]):
                mask = masks[0] if x.shape[2] in [32] else masks[1]
                mask = F.interpolate(mask, size=x.shape[2], mode='bilinear')
                x = x + self.hpf(mask * cache[x.shape[2]])
        out = self.to_out(x)
        return out

    def reset_parameters(self):
        self.apply(_reset_parameters)


class MappingNetwork(nn.Layer):
    def __init__(self,
                 latent_dim: int=16,
                 style_dim: int=48,
                 num_domains: int=2,
                 hidden_dim: int=384):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, hidden_dim)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(hidden_dim, hidden_dim)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.LayerList()
        for _ in range(num_domains):
            self.unshared.extend([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(), nn.Linear(hidden_dim, style_dim))
            ])

        self.reset_parameters()

    def forward(self, z: paddle.Tensor, y: paddle.Tensor):
        """Calculate forward propagation.
        Args:
            z(Tensor(float32)): 
                Shape (B, latent_dim).
            y(Tensor(float32)):
                speaker label. Shape (B, ).    
        Returns:
            Tensor:
                Shape (style_dim, )
        """
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        # (B, num_domains, style_dim)
        out = paddle.stack(out, axis=1)
        idx = paddle.arange(y.shape[0])
        # (style_dim, )
        s = out[idx, y]
        return s

    def reset_parameters(self):
        self.apply(_reset_parameters)


class StyleEncoder(nn.Layer):
    def __init__(self,
                 dim_in: int=48,
                 style_dim: int=48,
                 num_domains: int=2,
                 max_conv_dim: int=384):
        super().__init__()
        blocks = []
        blocks += [
            nn.Conv2D(
                in_channels=1,
                out_channels=dim_in,
                kernel_size=3,
                stride=1,
                padding=1)
        ]
        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [
                ResBlk(dim_in=dim_in, dim_out=dim_out, downsample='half')
            ]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [
            nn.Conv2D(
                in_channels=dim_out,
                out_channels=dim_out,
                kernel_size=5,
                stride=1,
                padding=0)
        ]
        blocks += [nn.AdaptiveAvgPool2D(1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)
        self.unshared = nn.LayerList()
        for _ in range(num_domains):
            self.unshared.append(nn.Linear(dim_out, style_dim))

        self.reset_parameters()

    def forward(self, x: paddle.Tensor, y: paddle.Tensor):
        """Calculate forward propagation.
        Args:
            x(Tensor(float32)): 
                Shape (B, 1, n_mels, T).   
            y(Tensor(float32)):
                speaker label. Shape (B, ).
        Returns:
            Tensor:
                Shape (style_dim, )
        """
        h = self.shared(x)
        h = h.reshape((h.shape[0], -1))
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        # (B, num_domains, style_dim)
        out = paddle.stack(out, axis=1)
        idx = paddle.arange(y.shape[0])
        # (style_dim,)
        s = out[idx, y]
        return s

    def reset_parameters(self):
        self.apply(_reset_parameters)


class Discriminator(nn.Layer):
    def __init__(self,
                 dim_in: int=48,
                 num_domains: int=2,
                 max_conv_dim: int=384,
                 repeat_num: int=4):
        super().__init__()
        # real/fake discriminator
        self.dis = Discriminator2D(
            dim_in=dim_in,
            num_domains=num_domains,
            max_conv_dim=max_conv_dim,
            repeat_num=repeat_num)
        # adversarial classifier
        self.cls = Discriminator2D(
            dim_in=dim_in,
            num_domains=num_domains,
            max_conv_dim=max_conv_dim,
            repeat_num=repeat_num)
        self.num_domains = num_domains

        self.reset_parameters()

    def forward(self, x: paddle.Tensor, y: paddle.Tensor):
        """Calculate forward propagation.
        Args:
            x(Tensor(float32)):
                Shape (B, 1, 80, T).
            y(Tensor(float32)):
                Shape (B, ). 
        Returns:
            Tensor:
                Shape (B, )
        """
        out = self.dis(x, y)
        return out

    def classifier(self, x: paddle.Tensor):
        out = self.cls.get_feature(x)
        return out

    def reset_parameters(self):
        self.apply(_reset_parameters)


class Discriminator2D(nn.Layer):
    def __init__(self,
                 dim_in: int=48,
                 num_domains: int=2,
                 max_conv_dim: int=384,
                 repeat_num: int=4):
        super().__init__()
        blocks = []
        blocks += [
            nn.Conv2D(
                in_channels=1,
                out_channels=dim_in,
                kernel_size=3,
                stride=1,
                padding=1)
        ]

        for lid in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [
            nn.Conv2D(
                in_channels=dim_out,
                out_channels=dim_out,
                kernel_size=5,
                stride=1,
                padding=0)
        ]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.AdaptiveAvgPool2D(1)]
        blocks += [
            nn.Conv2D(
                in_channels=dim_out,
                out_channels=num_domains,
                kernel_size=1,
                stride=1,
                padding=0)
        ]
        self.main = nn.Sequential(*blocks)

    def get_feature(self, x: paddle.Tensor):
        out = self.main(x)
        # (B, num_domains)
        out = out.reshape((out.shape[0], -1))
        return out

    def forward(self, x: paddle.Tensor, y: paddle.Tensor):
        out = self.get_feature(x)
        idx = paddle.arange(y.shape[0])
        # (B,) ?
        out = out[idx, y]
        return out
