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
import paddle
import paddle.nn.functional as F
from munch import Munch
from starganv2vc_paddle.transforms import build_transforms


# 这些都写到 updater 里
def compute_d_loss(nets,
                   args,
                   x_real,
                   y_org,
                   y_trg,
                   z_trg=None,
                   x_ref=None,
                   use_r1_reg=True,
                   use_adv_cls=False,
                   use_con_reg=False):
    args = Munch(args)

    assert (z_trg is None) != (x_ref is None)
    # with real audios
    x_real.stop_gradient = False
    out = nets.discriminator(x_real, y_org)
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
        out_aug = nets.discriminator(t(x_real).detach(), y_org)
        loss_con_reg += F.smooth_l1_loss(out, out_aug)

    # with fake audios
    with paddle.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)

        F0 = nets.f0_model.get_feature_GAN(x_real)
        x_fake = nets.generator(x_real, s_trg, masks=None, F0=F0)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)
    if use_con_reg:
        out_aug = nets.discriminator(t(x_fake).detach(), y_trg)
        loss_con_reg += F.smooth_l1_loss(out, out_aug)

    # adversarial classifier loss
    if use_adv_cls:
        out_de = nets.discriminator.classifier(x_fake)
        loss_real_adv_cls = F.cross_entropy(out_de[y_org != y_trg],
                                            y_org[y_org != y_trg])

        if use_con_reg:
            out_de_aug = nets.discriminator.classifier(t(x_fake).detach())
            loss_con_reg += F.smooth_l1_loss(out_de, out_de_aug)
    else:
        loss_real_adv_cls = paddle.zeros([1]).mean()

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg + \
            args.lambda_adv_cls * loss_real_adv_cls + \
            args.lambda_con_reg * loss_con_reg

    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item(),
                       real_adv_cls=loss_real_adv_cls.item(),
                       con_reg=loss_con_reg.item())


def compute_g_loss(nets,
                   args,
                   x_real,
                   y_org,
                   y_trg,
                   z_trgs=None,
                   x_refs=None,
                   use_adv_cls=False):
    args = Munch(args)

    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # compute style vectors
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)

    # compute ASR/F0 features (real)
    with paddle.no_grad():
        F0_real, GAN_F0_real, cyc_F0_real = nets.f0_model(x_real)
        ASR_real = nets.asr_model.get_feature(x_real)

    # adversarial loss
    x_fake = nets.generator(x_real, s_trg, masks=None, F0=GAN_F0_real)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # compute ASR/F0 features (fake)
    F0_fake, GAN_F0_fake, _ = nets.f0_model(x_fake)
    ASR_fake = nets.asr_model.get_feature(x_fake)

    # norm consistency loss
    x_fake_norm = log_norm(x_fake)
    x_real_norm = log_norm(x_real)
    loss_norm = ((paddle.nn.ReLU()(paddle.abs(x_fake_norm - x_real_norm) -
                                   args.norm_bias))**2).mean()

    # F0 loss
    loss_f0 = f0_loss(F0_fake, F0_real)

    # style F0 loss (style initialization)
    if x_refs is not None and args.lambda_f0_sty > 0 and not use_adv_cls:
        F0_sty, _, _ = nets.f0_model(x_ref)
        loss_f0_sty = F.l1_loss(compute_mean_f0(F0_fake),
                                compute_mean_f0(F0_sty))
    else:
        loss_f0_sty = paddle.zeros([1]).mean()

    # ASR loss
    loss_asr = F.smooth_l1_loss(ASR_fake, ASR_real)

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = paddle.mean(paddle.abs(s_pred - s_trg))

    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = nets.style_encoder(x_ref2, y_trg)
    x_fake2 = nets.generator(x_real, s_trg2, masks=None, F0=GAN_F0_real)
    x_fake2 = x_fake2.detach()
    _, GAN_F0_fake2, _ = nets.f0_model(x_fake2)
    loss_ds = paddle.mean(paddle.abs(x_fake - x_fake2))
    loss_ds += F.smooth_l1_loss(GAN_F0_fake, GAN_F0_fake2.detach())

    # cycle-consistency loss
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org, masks=None, F0=GAN_F0_fake)
    loss_cyc = paddle.mean(paddle.abs(x_rec - x_real))
    # F0 loss in cycle-consistency loss
    if args.lambda_f0 > 0:
        _, _, cyc_F0_rec = nets.f0_model(x_rec)
        loss_cyc += F.smooth_l1_loss(cyc_F0_rec, cyc_F0_real)
    if args.lambda_asr > 0:
        ASR_recon = nets.asr_model.get_feature(x_rec)
        loss_cyc += F.smooth_l1_loss(ASR_recon, ASR_real)

    # adversarial classifier loss
    if use_adv_cls:
        out_de = nets.discriminator.classifier(x_fake)
        loss_adv_cls = F.cross_entropy(out_de[y_org != y_trg],
                                       y_trg[y_org != y_trg])
    else:
        loss_adv_cls = paddle.zeros([1]).mean()

    loss = args.lambda_adv * loss_adv + args.lambda_sty * loss_sty \
           - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc\
           + args.lambda_norm * loss_norm \
           + args.lambda_asr * loss_asr \
           + args.lambda_f0 * loss_f0 \
           + args.lambda_f0_sty * loss_f0_sty \
           + args.lambda_adv_cls * loss_adv_cls

    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       ds=loss_ds.item(),
                       cyc=loss_cyc.item(),
                       norm=loss_norm.item(),
                       asr=loss_asr.item(),
                       f0=loss_f0.item(),
                       adv_cls=loss_adv_cls.item())


# for norm consistency loss
def log_norm(x, mean=-4, std=4, axis=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = paddle.log(paddle.exp(x * std + mean).norm(axis=axis))
    return x


# for adversarial loss
def adv_loss(logits, target):
    assert target in [1, 0]
    if len(logits.shape) > 1:
        logits = logits.reshape([-1])
    targets = paddle.full_like(logits, fill_value=target)
    logits = logits.clip(min=-10, max=10)  # prevent nan
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


# for R1 regularization loss
def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.shape[0]
    grad_dout = paddle.grad(outputs=d_out.sum(),
                            inputs=x_in,
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.shape == x_in.shape)
    reg = 0.5 * grad_dout2.reshape((batch_size, -1)).sum(1).mean(0)
    return reg


# for F0 consistency loss
def compute_mean_f0(f0):
    f0_mean = f0.mean(-1)
    f0_mean = f0_mean.expand((f0.shape[-1], f0_mean.shape[0])).transpose(
        (1, 0))  # (B, M)
    return f0_mean


def f0_loss(x_f0, y_f0):
    """
    x.shape = (B, 1, M, L): predict
    y.shape = (B, 1, M, L): target
    """
    # compute the mean
    x_mean = compute_mean_f0(x_f0)
    y_mean = compute_mean_f0(y_f0)
    loss = F.l1_loss(x_f0 / x_mean, y_f0 / y_mean)
    return loss
