"""
This file contains two PyTorch modules which together consist of the SEGAN model architecture
(based on the paper: Pascual et al. https://arxiv.org/pdf/1703.09452.pdf).
Modification of the initialization parameters allows the change of the model described in the class project,
such as turning the generator to a VAE, or removing the latent variable concatenation.

Loss functions for training SEGAN are also defined in this file.

Authors
 * Francis Carter 2021
"""

import paddle
import paddle.nn as nn
import paddle.utils.data
import paddle.nn.functional as F
from math import floor


class Generator(paddle.nn.Layer):
    """CNN Autoencoder model to clean speech signals.

    Arguments
    ---------
    kernel_size : int
        Size of the convolutional kernel.
    latent_vae : bool
        Whether or not to convert the autoencoder to a vae
    z_prob : bool
        Whether to remove the latent variable concatenation. Is only applicable if latent_vae is False
    """

    def __init__(self, kernel_size, latent_vae, z_prob):
        super().__init__()
        self.EncodeLayers = torch.nn.ModuleList()
        self.DecodeLayers = torch.nn.ModuleList()
        self.kernel_size = 5
        self.latent_vae = latent_vae
        self.z_prob = z_prob
        EncoderChannels = [1, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        DecoderChannels = [
            2048,
            1024,
            512,
            512,
            256,
            256,
            128,
            128,
            64,
            64,
            32,
            1,
        ]

        # Create encoder and decoder layers.
        for i in range(len(EncoderChannels) - 1):
            if i == len(EncoderChannels) - 2 and self.latent_vae:
                outs = EncoderChannels[i + 1] * 2
            else:
                outs = EncoderChannels[i + 1]
            self.EncodeLayers.append(
                nn.Conv1d(
                    in_channels=EncoderChannels[i],
                    out_channels=outs,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=floor(kernel_size / 2),  # same
                )
            )

        for i in range(len(DecoderChannels) - 1):
            if i == 0 and self.latent_vae:
                ins = EncoderChannels[-1 * (i + 1)]
            else:
                ins = EncoderChannels[-1 * (i + 1)] * 2
            self.DecodeLayers.append(
                nn.ConvTranspose1d(
                    in_channels=ins,
                    out_channels=EncoderChannels[-1 * (i + 2)],
                    kernel_size=kernel_size
                    + 1,  # adding one to kernel size makes the dimensions match
                    stride=2,
                    padding=floor(kernel_size / 2),  # same
                )
            )

    def forward(self, x):
        """Forward pass through autoencoder"""
        # encode
        skips = []
        x = x.permute(0, 2, 1)
        for i, layer in enumerate(self.EncodeLayers):
            x = layer(x)
            skips.append(x.clone())
            if i == len(self.DecodeLayers) - 1:
                continue
            else:
                x = F.leaky_relu(x, negative_slope=0.3)

        # fuse z
        if self.latent_vae:
            z_mean, z_logvar = x.chunk(2, dim=1)
            x = z_mean + torch.exp(z_logvar / 2.0) * torch.randn_like(
                z_logvar, device=x.device
            )  # sampling from latent var probability distribution
        elif self.z_prob:
            z = torch.normal(torch.zeros_like(x), torch.ones_like(x))
            x = torch.cat((x, z), 1)
        else:
            z = torch.zeros_like(x)
            x = torch.cat((x, z), 1)

        # decode
        for i, layer in enumerate(self.DecodeLayers):
            x = layer(x)
            if i == len(self.DecodeLayers) - 1:
                continue
            else:
                x = torch.cat((x, skips[-1 * (i + 2)]), 1)
                x = F.leaky_relu(x, negative_slope=0.3)
        x = x.permute(0, 2, 1)
        if self.latent_vae:
            return x, z_mean, z_logvar
        else:
            return x


class Discriminator(paddle.nn.Layer):
    """CNN discriminator of SEGAN

    Arguments
    ---------
    kernel_size : int
        Size of the convolutional kernel.
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.Layers = torch.nn.ModuleList()
        self.Norms = torch.nn.ModuleList()
        Channels = [2, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024, 1]
        # Create encoder and decoder layers.
        for i in range(len(Channels) - 1):
            if i != len(Channels) - 2:
                self.Layers.append(
                    nn.Conv1d(
                        in_channels=Channels[i],
                        out_channels=Channels[i + 1],
                        kernel_size=kernel_size,
                        stride=2,
                        padding=floor(kernel_size / 2),  # same
                    )
                )
                self.Norms.append(
                    nn.BatchNorm1d(
                        num_features=Channels[
                            i + 1
                        ]  # not sure what the last dim should be here
                    )
                )
            # output convolution
            else:
                self.Layers.append(
                    nn.Conv1d(
                        in_channels=Channels[i],
                        out_channels=Channels[i + 1],
                        kernel_size=1,
                        stride=1,
                        padding=0,  # same
                    )
                )
                self.Layers.append(
                    nn.Linear(in_features=8, out_features=1,)  # Channels[i+1],
                )

    def forward(self, x):
        """forward pass through the discriminator"""
        x = x.permute(0, 2, 1)
        # encode
        for i in range(len(self.Norms)):
            x = self.Layers[i](x)
            x = self.Norms[i](x)
            x = F.leaky_relu(x, negative_slope=0.3)

        # output
        x = self.Layers[-2](x)
        x = self.Layers[-1](x)
        # x = F.sigmoid(x)
        x = x.permute(0, 2, 1)

        return x  # in logit format


def d1_loss(d_outputs, reduction="mean"):
    """Calculates the loss of the discriminator when the inputs are clean    """
    output = 0.5 * ((d_outputs - 1) ** 2)
    if reduction == "mean":
        return output.mean()
    elif reduction == "batch":
        return output.view(output.size(0), -1).mean(1)


def d2_loss(d_outputs, reduction="mean"):
    """Calculates the loss of the discriminator when the inputs are not clean    """
    output = 0.5 * ((d_outputs) ** 2)
    if reduction == "mean":
        return output.mean()
    elif reduction == "batch":
        return output.view(output.size(0), -1).mean(1)


def g3_loss(
    d_outputs,
    predictions,
    targets,
    length,
    l1LossCoeff,
    klLossCoeff,
    z_mean=None,
    z_logvar=None,
    reduction="mean",
):
    """Calculates the loss of the generator given the discriminator outputs    """
    discrimloss = 0.5 * ((d_outputs - 1) ** 2)
    l1norm = torch.nn.functional.l1_loss(predictions, targets, reduction="none")

    if not (
        z_mean is None
    ):  # This will determine if model is being trained as a vae
        ZERO = torch.zeros_like(z_mean)
        distq = torch.distributions.normal.Normal(
            z_mean, torch.exp(z_logvar) ** (1 / 2)
        )
        distp = torch.distributions.normal.Normal(
            ZERO, torch.exp(ZERO) ** (1 / 2)
        )
        kl = torch.distributions.kl.kl_divergence(distq, distp)
        kl = kl.sum(axis=1).sum(axis=1).mean()
    else:
        kl = 0
    if reduction == "mean":
        return (
            discrimloss.mean() + l1LossCoeff * l1norm.mean() + klLossCoeff * kl
        )
    elif reduction == "batch":
        dloss = discrimloss.view(discrimloss.size(0), -1).mean(1)
        lloss = l1norm.view(l1norm.size(0), -1).mean(1)
        return dloss + l1LossCoeff * lloss + klLossCoeff * kl
