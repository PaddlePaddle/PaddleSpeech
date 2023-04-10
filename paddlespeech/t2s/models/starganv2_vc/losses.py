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
from typing import Any
from typing import Dict

import paddle
import paddle.nn.functional as F

from .transforms import build_transforms


# 这些都写到 updater 里
def compute_d_loss(nets: Dict[str, Any],
                   x_real: paddle.Tensor,
                   y_org: paddle.Tensor,
                   y_trg: paddle.Tensor,
                   z_trg: paddle.Tensor=None,
                   x_ref: paddle.Tensor=None,
                   use_r1_reg=True,
                   use_adv_cls=False,
                   use_con_reg=False,
                   lambda_reg: float=1.,
                   lambda_adv_cls: float=0.1,
                   lambda_con_reg: float=10.):

    assert (z_trg is None) != (x_ref is None)
    # with real audios
    x_real.stop_gradient = False

    out = nets['discriminator'](x_real, y_org)
    loss_real = adv_loss(out, 1)

    # R1 regularizaition (https://arxiv.org/abs/1801.04406v4)
    if use_r1_reg:
        loss_reg = r1_reg(out, x_real)
    else:
        loss_reg = paddle.to_tensor([0.], dtype=paddle.float32)

    # consistency regularization (bCR-GAN: https://arxiv.org/abs/2002.04724)
    loss_con_reg = paddle.to_tensor([0.], dtype=paddle.float32)
    if use_con_reg:
        t = build_transforms()
        out_aug = nets['discriminator'](t(x_real).detach(), y_org)
        loss_con_reg += F.smooth_l1_loss(out, out_aug)

    # with fake audios
    with paddle.no_grad():
        if z_trg is not None:
            s_trg = nets['mapping_network'](z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets['style_encoder'](x_ref, y_trg)

        F0 = nets['F0_model'].get_feature_GAN(x_real)
        x_fake = nets['generator'](x_real, s_trg, masks=None, F0=F0)
    out = nets['discriminator'](x_fake, y_trg)
    loss_fake = adv_loss(out, 0)
    if use_con_reg:
        out_aug = nets['discriminator'](t(x_fake).detach(), y_trg)
        loss_con_reg += F.smooth_l1_loss(out, out_aug)

    # adversarial classifier loss
    if use_adv_cls:
        out_de = nets['discriminator'].classifier(x_fake)
        loss_real_adv_cls = F.cross_entropy(out_de[y_org != y_trg],
                                            y_org[y_org != y_trg])

        if use_con_reg:
            out_de_aug = nets['discriminator'].classifier(t(x_fake).detach())
            loss_con_reg += F.smooth_l1_loss(out_de, out_de_aug)
    else:
        loss_real_adv_cls = paddle.zeros([1]).mean()

    loss = loss_real + loss_fake + lambda_reg * loss_reg + \
            lambda_adv_cls * loss_real_adv_cls + \
            lambda_con_reg * loss_con_reg

    return loss


def compute_g_loss(nets: Dict[str, Any],
                   x_real: paddle.Tensor,
                   y_org: paddle.Tensor,
                   y_trg: paddle.Tensor,
                   z_trgs: paddle.Tensor=None,
                   x_refs: paddle.Tensor=None,
                   use_adv_cls: bool=False,
                   lambda_sty: float=1.,
                   lambda_cyc: float=5.,
                   lambda_ds: float=1.,
                   lambda_norm: float=1.,
                   lambda_asr: float=10.,
                   lambda_f0: float=5.,
                   lambda_f0_sty: float=0.1,
                   lambda_adv: float=2.,
                   lambda_adv_cls: float=0.5,
                   norm_bias: float=0.5):

    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # compute style vectors
    if z_trgs is not None:
        s_trg = nets['mapping_network'](z_trg, y_trg)
    else:
        s_trg = nets['style_encoder'](x_ref, y_trg)

    # compute ASR/F0 features (real)
    with paddle.no_grad():
        F0_real, GAN_F0_real, cyc_F0_real = nets['F0_model'](x_real)
        ASR_real = nets['asr_model'].get_feature(x_real)

    # adversarial loss
    x_fake = nets['generator'](x_real, s_trg, masks=None, F0=GAN_F0_real)
    out = nets['discriminator'](x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # compute ASR/F0 features (fake)
    F0_fake, GAN_F0_fake, _ = nets['F0_model'](x_fake)
    ASR_fake = nets['asr_model'].get_feature(x_fake)

    # norm consistency loss
    x_fake_norm = log_norm(x_fake)
    x_real_norm = log_norm(x_real)
    tmp = paddle.abs(x_fake_norm - x_real_norm) - norm_bias
    loss_norm = ((paddle.nn.ReLU()(tmp))**2).mean()

    # F0 loss
    loss_f0 = f0_loss(F0_fake, F0_real)

    # style F0 loss (style initialization)
    if x_refs is not None and lambda_f0_sty > 0 and not use_adv_cls:
        F0_sty, _, _ = nets['F0_model'](x_ref)
        loss_f0_sty = F.l1_loss(
            compute_mean_f0(F0_fake), compute_mean_f0(F0_sty))
    else:
        loss_f0_sty = paddle.zeros([1]).mean()

    # ASR loss
    loss_asr = F.smooth_l1_loss(ASR_fake, ASR_real)

    # style reconstruction loss
    s_pred = nets['style_encoder'](x_fake, y_trg)
    loss_sty = paddle.mean(paddle.abs(s_pred - s_trg))

    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets['mapping_network'](z_trg2, y_trg)
    else:
        s_trg2 = nets['style_encoder'](x_ref2, y_trg)
    x_fake2 = nets['generator'](x_real, s_trg2, masks=None, F0=GAN_F0_real)
    x_fake2 = x_fake2.detach()
    _, GAN_F0_fake2, _ = nets['F0_model'](x_fake2)
    loss_ds = paddle.mean(paddle.abs(x_fake - x_fake2))
    loss_ds += F.smooth_l1_loss(GAN_F0_fake, GAN_F0_fake2.detach())

    # cycle-consistency loss
    s_org = nets['style_encoder'](x_real, y_org)
    x_rec = nets['generator'](x_fake, s_org, masks=None, F0=GAN_F0_fake)
    loss_cyc = paddle.mean(paddle.abs(x_rec - x_real))
    # F0 loss in cycle-consistency loss
    if lambda_f0 > 0:
        _, _, cyc_F0_rec = nets['F0_model'](x_rec)
        loss_cyc += F.smooth_l1_loss(cyc_F0_rec, cyc_F0_real)
    if lambda_asr > 0:
        ASR_recon = nets['asr_model'].get_feature(x_rec)
        loss_cyc += F.smooth_l1_loss(ASR_recon, ASR_real)

    # adversarial classifier loss
    if use_adv_cls:
        out_de = nets['discriminator'].classifier(x_fake)
        loss_adv_cls = F.cross_entropy(out_de[y_org != y_trg],
                                       y_trg[y_org != y_trg])
    else:
        loss_adv_cls = paddle.zeros([1]).mean()

    loss = lambda_adv * loss_adv + lambda_sty * loss_sty \
           - lambda_ds * loss_ds + lambda_cyc * loss_cyc \
           + lambda_norm * loss_norm \
           + lambda_asr * loss_asr \
           + lambda_f0 * loss_f0 \
           + lambda_f0_sty * loss_f0_sty \
           + lambda_adv_cls * loss_adv_cls

    return loss


# for norm consistency loss
def log_norm(x: paddle.Tensor, mean: float=-4, std: float=4, axis: int=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = paddle.log(paddle.exp(x * std + mean).norm(axis=axis))
    return x


# for adversarial loss
def adv_loss(logits: paddle.Tensor, target: float):
    assert target in [1, 0]
    if len(logits.shape) > 1:
        logits = logits.reshape([-1])
    targets = paddle.full_like(logits, fill_value=target)
    logits = logits.clip(min=-10, max=10)  # prevent nan
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


# for R1 regularization loss
def r1_reg(d_out: paddle.Tensor, x_in: paddle.Tensor):
    # zero-centered gradient penalty for real images
    batch_size = x_in.shape[0]
    grad_dout = paddle.grad(
        outputs=d_out.sum(),
        inputs=x_in,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.shape == x_in.shape)
    reg = 0.5 * grad_dout2.reshape((batch_size, -1)).sum(1).mean(0)
    return reg


# for F0 consistency loss
def compute_mean_f0(f0: paddle.Tensor):
    f0_mean = f0.mean(-1)
    f0_mean = f0_mean.expand((f0.shape[-1], f0_mean.shape[0])).transpose(
        (1, 0))  # (B, M)
    return f0_mean


def f0_loss(x_f0: paddle.Tensor, y_f0: paddle.Tensor):
    """
    x.shape = (B, 1, M, L): predict
    y.shape = (B, 1, M, L): target
    """
    # compute the mean
    x_mean = compute_mean_f0(x_f0)
    y_mean = compute_mean_f0(y_f0)
    loss = F.l1_loss(x_f0 / x_mean, y_f0 / y_mean)
    return loss
