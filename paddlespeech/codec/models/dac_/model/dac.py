from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn

from paddlespeech.audiotools import AudioSignal
from paddlespeech.codec.models.dac_.model.base import CodecMixin
from paddlespeech.codec.models.dac_.nn.layers import Snake1d
from paddlespeech.codec.models.dac_.nn.layers import WNConv1d
from paddlespeech.codec.models.dac_.nn.layers import WNConvTranspose1d
from paddlespeech.codec.models.dac_.nn.quantize import ResidualVectorQuantize


def init_weights(m):
    if isinstance(m, nn.Conv1D):
        nn.initializer.TruncatedNormal(std=0.02)(m.weight)
        nn.initializer.Constant(0)(m.bias)


class ResidualUnit(nn.Layer):
    def __init__(self, dim: int=16, dilation: int=1):
        super(ResidualUnit, self).__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1), )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Layer):
    def __init__(self, dim: int=16, stride: int=1):
        super(EncoderBlock, self).__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=int(np.ceil(stride / 2).astype(int)), ), )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Layer):
    def __init__(
            self,
            d_model: int=64,
            strides: list=[2, 4, 8, 8],
            d_latent: int=64, ):
        super(Encoder, self).__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Layer):
    def __init__(self, input_dim: int=16, output_dim: int=8, stride: int=1):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=int(np.ceil(stride / 2).astype(int)), ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9), )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Layer):
    def __init__(
            self,
            input_channel,
            channels,
            rates,
            d_out: int=1, ):
        super(Decoder, self).__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2**(i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DAC(nn.Layer, CodecMixin):
    def __init__(
            self,
            encoder_dim: int=64,
            encoder_rates: list=[2, 4, 8, 8],
            latent_dim: int=None,
            decoder_dim: int=1536,
            decoder_rates: list=[8, 8, 4, 2],
            n_codebooks: int=9,
            codebook_size: int=1024,
            codebook_dim: int=8,
            quantizer_dropout: bool=False,
            sample_rate: int=44100, ):
        super(DAC, self).__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2**len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout, )

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates, )
        self.sample_rate = sample_rate
        self.apply(init_weights)

        self.delay = self.get_delay()

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = np.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = F.pad(
            audio_data, [0, int(right_pad)],
            mode='constant',
            value=0,
            data_format="NCL")

        return audio_data

    def encode(
            self,
            audio_data: paddle.Tensor,
            n_quantizers: int=None, ):
        """Encode given audio data and return quantized latent codes

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        n_quantizers : int, optional
            Number of quantizers to use, by default None
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
        """
        z = self.encoder(audio_data)
        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(
            z, n_quantizers)
        return z, codes, latents, commitment_loss, codebook_loss

    def decode(self, z: paddle.Tensor):
        """Decode given latent codes and return audio data

        Parameters
        ----------
        z : Tensor[B x D x T]
            Quantized continuous representation of input
        length : int, optional
            Number of samples in output audio, by default None

        Returns
        -------
        dict
            A dictionary with the following keys:
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        return self.decoder(z)

    def forward(
            self,
            audio_data: paddle.Tensor,
            sample_rate: int=None,
            n_quantizers: int=None, ):
        """Model forward pass

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`
        n_quantizers : int, optional
            Number of quantizers to use, by default None.
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        z, codes, latents, commitment_loss, codebook_loss = self.encode(
            audio_data, n_quantizers)

        x = self.decode(z)
        return {
            "audio": x[..., :length],
            "z": z,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }


if __name__ == "__main__":

    model = DAC()

    def fn(o, p):
        return o + f" {p / 1e6:<.3f}M params."

    for n, m in model.named_sublayers():
        # print(type(m))
        o = m.extra_repr()
        p = sum([np.prod(p.shape) for p in m.parameters()])
        setattr(m, "extra_repr", partial(fn, o=o, p=p))
    print(model)
    print("Total # of params: ",
          sum([np.prod(p.shape) for p in model.parameters()]))

    length = 88200 * 2
    x = paddle.randn([1, 1, length])
    x.stop_gradient = False

    # Make a forward pass
    out = model(x)["audio"]
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)

    # Create gradient variable
    grad = paddle.zeros_like(out)
    grad[:, :, grad.shape[-1] // 2] = 1

    # Make a backward pass
    out.backward(grad)

    # Check non-zero values
    gradmap = x.grad.squeeze(0)
    gradmap = (gradmap != 0).sum(0)  # sum across features
    rf = (gradmap != 0).sum()

    print(f"Receptive field: {rf.item()}"
          )  # TODO: drryanhuang, fix this question, why is RF zero?

    x = AudioSignal(paddle.randn([1, 1, 44100 * 60]), 44100)
    model.decompress(model.compress(x, verbose=True), verbose=True)
