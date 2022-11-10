# MIT License, Copyright (c) 2022 OpenAI.
# Copyright (c) 2022 PaddlePaddle Authors and . All Rights Reserved.
# 
# Modified from OpenAI Whisper 2022 (https://github.com/openai/whisper/whisper/model.py)
from dataclasses import dataclass
from typing import Dict
from typing import Iterable
from typing import Optional

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn

import paddlespeech.s2t.modules.align as paddlespeech_nn
from paddlespeech.s2t.models.whisper.decoding import decode as decode_function
from paddlespeech.s2t.models.whisper.decoding import detect_language as detect_language_function
from paddlespeech.s2t.models.whisper.transcribe import transcribe as transcribe_function
#paddle 中的nn init和torch不对齐，训练的时候会有问题


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(paddlespeech_nn.LayerNorm):
    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return super().forward(x)


class Linear(paddlespeech_nn.Linear):
    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return F.linear(x, self.weight, None
                        if self.bias is None else self.bias)


class Conv1d(paddlespeech_nn.Conv1D):
    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return super().forward(x)


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = paddle.exp(-log_timescale_increment * paddle.arange(
        channels // 2, dtype=paddle.float32))
    scaled_time = paddle.arange(
        length,
        dtype=paddle.float32)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return paddle.to_tensor(
        paddle.concat(
            [paddle.sin(scaled_time), paddle.cos(scaled_time)], axis=1))


class MultiHeadAttention(nn.Layer):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state, bias_attr=True)
        self.key = Linear(n_state, n_state, bias_attr=False)
        self.value = Linear(n_state, n_state, bias_attr=True)
        self.out = Linear(n_state, n_state, bias_attr=True)

    def forward(
            self,
            x: paddle.Tensor,
            xa: Optional[paddle.Tensor]=None,
            mask: Optional[paddle.Tensor]=None,
            kv_cache: Optional[dict]=None, ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv = self.qkv_attention(q, k, v, mask)
        return self.out(wv)

    def qkv_attention(self,
                      q: paddle.Tensor,
                      k: paddle.Tensor,
                      v: paddle.Tensor,
                      mask: Optional[paddle.Tensor]=None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head)**-0.25
        q = paddle.transpose(
            q.view(*q.shape[:2], self.n_head, -1), (0, 2, 1, 3)) * scale
        k = paddle.transpose(
            k.view(*k.shape[:2], self.n_head, -1), (0, 2, 3, 1)) * scale
        v = paddle.transpose(
            v.view(*v.shape[:2], self.n_head, -1), (0, 2, 1, 3))

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]

        w = F.softmax(qk.float(), axis=-1).to(q.dtype)
        return paddle.transpose((w @ v), (0, 2, 1, 3)).flatten(start_axis=2)


class ResidualAttentionBlock(nn.Layer):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool=False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(
            n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp, bias_attr=True),
            nn.GELU(), Linear(n_mlp, n_state, bias_attr=True))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
            self,
            x: paddle.Tensor,
            xa: Optional[paddle.Tensor]=None,
            mask: Optional[paddle.Tensor]=None,
            kv_cache: Optional[dict]=None, ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
        if self.cross_attn:
            x = x + self.cross_attn(
                self.cross_attn_ln(x), xa, kv_cache=kv_cache)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Layer):
    def __init__(self,
                 n_mels: int,
                 n_ctx: int,
                 n_state: int,
                 n_head: int,
                 n_layer: int):
        super().__init__()
        self.conv1 = Conv1d(
            n_mels, n_state, kernel_size=3, stride=1, padding=1, bias_attr=True)
        self.conv2 = Conv1d(
            n_state,
            n_state,
            kernel_size=3,
            stride=2,
            padding=1,
            bias_attr=True)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.LayerList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)])
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: paddle.Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = paddle.transpose(x, (0, 2, 1))

        assert x.shape[
            1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Layer):
    def __init__(self,
                 n_vocab: int,
                 n_ctx: int,
                 n_state: int,
                 n_head: int,
                 n_layer: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = paddle.create_parameter(
            shape=[n_ctx, n_state], dtype='float32')

        self.blocks: Iterable[ResidualAttentionBlock] = nn.LayerList([
            ResidualAttentionBlock(n_state, n_head, cross_attention=True)
            for _ in range(n_layer)
        ])
        self.ln = LayerNorm(n_state)

        mask = fluid.layers.fill_constant(
            shape=[n_ctx, n_state], value=-np.inf, dtype='float32')
        mask = paddle.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistable=False)

    def forward(self,
                x: paddle.Tensor,
                xa: paddle.Tensor,
                kv_cache: Optional[dict]=None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = self.token_embedding(x) + self.positional_embedding[offset:offset +
                                                                x.shape[-1]]
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (x @ paddle.transpose(self.token_embedding.weight, (1, 0)))

        return logits


class Whisper(nn.Layer):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer, )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer, )

    def embed_audio(self, mel: paddle.Tensor):
        return self.encoder.forward(mel)

    def logits(self, tokens: paddle.Tensor, audio_features: paddle.Tensor):
        return self.decoder.forward(tokens, audio_features)

    def forward(self, mel: paddle.Tensor,
                tokens: paddle.Tensor) -> Dict[str, paddle.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return paddle.device.get_device()
        #return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict]=None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[
                    1] > self.decoder.positional_embedding.shape[0]:
                cache[
                    module] = output  # save as-is, for the first token or cross attention
            else:
                cache[module] = paddle.concat(
                    [cache[module], output], axis=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Layer):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(
                    layer.key.register_forward_post_hook(save_to_cache))
                hooks.append(
                    layer.value.register_forward_post_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function
