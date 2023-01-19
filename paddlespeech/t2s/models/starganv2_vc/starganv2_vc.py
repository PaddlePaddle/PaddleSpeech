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
# import copy
import math

import paddle
import paddle.nn.functional as F
from paddle import nn

from paddlespeech.utils.initialize import _calculate_gain
from paddlespeech.utils.initialize import xavier_uniform_

# from munch import Munch


class DownSample(nn.Layer):
    def __init__(self, layer_type: str):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == 'half':
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError(
                'Got unexpected donwsampletype %s, expected is [none, timepreserve, half]'
                % self.layer_type)


class UpSample(nn.Layer):
    def __init__(self, layer_type: str):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.interpolate(x, scale_factor=(2, 1), mode='nearest')
        elif self.layer_type == 'half':
            return F.interpolate(x, scale_factor=2, mode='nearest')
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
        x = self._shortcut(x) + self._residual(x)
        # unit variance
        return x / math.sqrt(2)


class AdaIN(nn.Layer):
    def __init__(self, style_dim: int, num_features: int):
        super().__init__()
        self.norm = nn.InstanceNorm2D(
            num_features=num_features, weight_attr=False, bias_attr=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x: paddle.Tensor, s: paddle.Tensor):
        if len(s.shape) == 1:
            s = s[None]
        h = self.fc(s)
        h = h.reshape((h.shape[0], h.shape[1], 1, 1))
        gamma, beta = paddle.split(h, 2, axis=1)
        return (1 + gamma) * self.norm(x) + beta


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
        return F.conv2d(x, filter, padding=1, groups=x.shape[1])


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

    def forward(self,
                x: paddle.Tensor,
                s: paddle.Tensor,
                masks: paddle.Tensor=None,
                F0: paddle.Tensor=None):
        x = self.stem(x)
        cache = {}
        for block in self.encode:
            if (masks is not None) and (x.shape[2] in [32, 64, 128]):
                cache[x.shape[2]] = x
            x = block(x)

        if F0 is not None:
            F0 = self.F0_conv(F0)
            F0 = F.adaptive_avg_pool2d(F0, [x.shape[-2], x.shape[-1]])
            x = paddle.concat([x, F0], axis=1)

        for block in self.decode:
            x = block(x, s)
            if (masks is not None) and (x.shape[2] in [32, 64, 128]):
                mask = masks[0] if x.shape[2] in [32] else masks[1]
                mask = F.interpolate(mask, size=x.shape[2], mode='bilinear')
                x = x + self.hpf(mask * cache[x.shape[2]])

        return self.to_out(x)


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

    def forward(self, z: paddle.Tensor, y: paddle.Tensor):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        # (batch, num_domains, style_dim)
        out = paddle.stack(out, axis=1)
        idx = paddle.arange(y.shape[0])
        # (batch, style_dim)
        s = out[idx, y]
        return s


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

    def forward(self, x: paddle.Tensor, y: paddle.Tensor):
        h = self.shared(x)
        h = h.reshape((h.shape[0], -1))
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        # (batch, num_domains, style_dim)
        out = paddle.stack(out, axis=1)
        idx = paddle.arange(y.shape[0])
        # (batch, style_dim)
        s = out[idx, y]
        return s


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

    def forward(self, x: paddle.Tensor, y: paddle.Tensor):
        return self.dis(x, y)

    def classifier(self, x: paddle.Tensor):
        return self.cls.get_feature(x)


class LinearNorm(nn.Layer):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 bias: bool=True,
                 w_init_gain: str='linear'):
        super().__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias_attr=bias)
        xavier_uniform_(
            self.linear_layer.weight, gain=_calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


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
        # (batch, num_domains)
        out = out.reshape((out.shape[0], -1))
        return out

    def forward(self, x: paddle.Tensor, y: paddle.Tensor):
        out = self.get_feature(x)
        idx = paddle.arange(y.shape[0])
        # (batch)
        out = out[idx, y]
        return out


'''
def build_model(args, F0_model: nn.Layer, ASR_model: nn.Layer):
    generator = Generator(
        dim_in=args.dim_in,
        style_dim=args.style_dim,
        max_conv_dim=args.max_conv_dim,
        w_hpf=args.w_hpf,
        F0_channel=args.F0_channel)
    mapping_network = MappingNetwork(
        latent_dim=args.latent_dim,
        style_dim=args.style_dim,
        num_domains=args.num_domains,
        hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(
        dim_in=args.dim_in,
        style_dim=args.style_dim,
        num_domains=args.num_domains,
        max_conv_dim=args.max_conv_dim)
    discriminator = Discriminator(
        dim_in=args.dim_in,
        num_domains=args.num_domains,
        max_conv_dim=args.max_conv_dim,
        n_repeat=args.n_repeat)
    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)

    nets = Munch(
        generator=generator,
        mapping_network=mapping_network,
        style_encoder=style_encoder,
        discriminator=discriminator,
        f0_model=F0_model,
        asr_model=ASR_model)

    nets_ema = Munch(
        generator=generator_ema,
        mapping_network=mapping_network_ema,
        style_encoder=style_encoder_ema)

    return nets, nets_ema


class StarGANv2VC(nn.Layer):
    def __init__(
            self,
            # spk_num
            num_domains: int=20,
            dim_in: int=64,
            style_dim: int=64,
            latent_dim: int=16,
            max_conv_dim: int=512,
            n_repeat: int=4,
            w_hpf: int=0,
            F0_channel: int=256):
        super().__init__()

        self.generator = Generator(
            dim_in=dim_in,
            style_dim=style_dim,
            max_conv_dim=max_conv_dim,
            w_hpf=w_hpf,
            F0_channel=F0_channel)
        # MappingNetwork and StyleEncoder are used to generate reference_embeddings
        self.mapping_network = MappingNetwork(
            latent_dim=latent_dim,
            style_dim=style_dim,
            num_domains=num_domains,
            hidden_dim=max_conv_dim)

        self.style_encoder = StyleEncoder(
            dim_in=dim_in,
            style_dim=style_dim,
            num_domains=num_domains,
            max_conv_dim=max_conv_dim)

        self.discriminator = Discriminator(
            dim_in=dim_in,
            num_domains=num_domains,
            max_conv_dim=max_conv_dim,
            repeat_num=n_repeat)
'''
