from typing import Union

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.utils import weight_norm

from paddlespeech.codec.models.dac_.nn.layers import WNConv1d


class VectorQuantize(nn.Layer):
    def __init__(self, input_dim, codebook_size, codebook_dim):
        super(VectorQuantize, self).__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z):
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        z_q, indices = self.decode_latents(z_e)

        commitment_loss = F.mse_loss(
            z_e, z_q.detach(), reduction='none').mean(axis=[1, 2])
        codebook_loss = F.mse_loss(
            z_q, z_e.detach(), reduction='none').mean(axis=[1, 2])

        z_q = z_e + (z_q - z_e).detach(
        )  # noop in forward pass, straight-through gradient estimator in backward pass
        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose([0, 2, 1])

    def decode_latents(self, latents):
        # encodings = Rearrange('b d t -> (b t) d')(latents)
        encodings = latents.transpose([0, 2, 1]).flatten(stop_axis=1)
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings, axis=1)
        codebook = F.normalize(codebook, axis=1)

        # Compute euclidean distance with codebook
        dist = (encodings.pow(2).sum(axis=1, keepdim=True) - 2 * encodings
                @ codebook.T + codebook.pow(2).sum(axis=1, keepdim=True).T)
        # indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        indices = (-dist).argmax(axis=1).reshape([latents.shape[0], -1])
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(nn.Layer):
    def __init__(
            self,
            input_dim: int=512,
            n_codebooks: int=9,
            codebook_size: int=1024,
            codebook_dim: Union[int, list]=8,
            quantizer_dropout: float=0.0, ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.LayerList([
            VectorQuantize(input_dim, codebook_size, codebook_dim[i])
            for i in range(n_codebooks)
        ])
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z, n_quantizers: int=None):
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = paddle.ones((z.shape[0], )) * self.n_codebooks + 1
            dropout = paddle.randint(1, self.n_codebooks + 1, (z.shape[0], ))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            if dropout[:n_dropout].size:
                n_quantizers[:n_dropout] = dropout[:n_dropout]

        for i, quantizer in enumerate(self.quantizers):
            if not self.training and i >= n_quantizers:
                break

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                residual)

            mask = (paddle.full(
                (z.shape[0], ), fill_value=i) < n_quantizers).astype("float32")
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = paddle.stack(codebook_indices, axis=1)
        latents = paddle.concat(latents, axis=1)

        return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: paddle.Tensor):
        z_q = 0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, paddle.concat(z_p, axis=1), codes

    def from_latents(self, latents: paddle.Tensor):
        z_q = 0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(
            axis=0, keepdims=True)[0]
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(
                latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, paddle.concat(z_p, axis=1), paddle.stack(codes, axis=1)


if __name__ == "__main__":
    rvq = ResidualVectorQuantize(quantizer_dropout=True)
    x = paddle.randn((16, 512, 80))
    y = rvq(x)
    print(y[2].shape)
