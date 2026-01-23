# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
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
from typing import Tuple, Any, Dict, Optional
import paddle
import math
from paddle import nn
import paddle.nn.functional as F
from einops import pack, rearrange, repeat
from paddlespeech.t2s.models.CosyVoice.common import mask_to_bias
from paddlespeech.t2s.models.CosyVoice.mask import add_optional_chunk_mask
from .matcha_transformer import BasicTransformerBlock

def get_activation(act_fn):
    if act_fn == "silu":
        return nn.Silu()
    elif act_fn == "mish":
        return nn.Mish()
    elif act_fn == "relu":
        return nn.ReLU()
    elif act_fn == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")

class Block1D(nn.Layer):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1D(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.Mish(),
        )

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask

class ResnetBlock1D(nn.Layer):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Mish(), 
            nn.Linear(time_emb_dim, dim_out)
        )

        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1D(dim, dim_out, 1)

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        # 添加时间嵌入并调整维度
        h += self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output

class Downsample1D(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1D(dim, dim, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class TimestepEmbedding(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None and self.cond_proj is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)
        if self.act is not None:
            sample = self.act(sample)
        sample = self.linear_2(sample)
        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample

class Upsample1D(nn.Layer):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
    """

    def __init__(self, channels, use_conv=False, use_conv_transpose=True, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        self.conv = None
        if use_conv_transpose:
            self.conv = nn.Conv1DTranspose(channels, self.out_channels, 4, stride=2, padding=1)
        elif use_conv:
            self.conv = nn.Conv1D(self.channels, self.out_channels, 3, padding=1)

    def forward(self, inputs):
        assert inputs.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(inputs)

        outputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")

        if self.use_conv:
            outputs = self.conv(outputs)

        return outputs

class Transpose(nn.Layer):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = paddle.transpose(x, [0, self.dim1, self.dim0])
        return x

class CausalConv1d(nn.Conv1D):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = 'zeros'
    ) -> None:
        super(CausalConv1d, self).__init__(in_channels, out_channels,
                                           kernel_size, stride,
                                           padding=0, dilation=dilation,
                                           groups=groups,
                                           padding_mode=padding_mode)
        assert stride == 1
        self.causal_padding = kernel_size - 1

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = F.pad(x, (self.causal_padding, 0), value=0.0)
        x = super(CausalConv1d, self).forward(x)
        return x

class SinusoidalPosEmb(paddle.nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"
    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = paddle.exp(paddle.arange(half_dim).astype('float32') * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = paddle.concat([paddle.sin(emb), paddle.cos(emb)], axis=-1)
        return emb

class CausalBlock1D(Block1D):
    def __init__(self, dim: int, dim_out: int):
        super(CausalBlock1D, self).__init__(dim, dim_out)
        self.block = nn.Sequential(
            CausalConv1d(dim, dim_out, 3),
            Transpose(1, 2),
            nn.LayerNorm(dim_out),
            Transpose(1, 2), 
            nn.Mish()                     
        )

    def forward(self, x: paddle.Tensor, mask: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
        output = self.block(x * mask)
        return output * mask


class CausalResnetBlock1D(ResnetBlock1D):
    def __init__(self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8):
        super(CausalResnetBlock1D, self).__init__(dim, dim_out, time_emb_dim, groups)
        self.block1 = CausalBlock1D(dim, dim_out)
        self.block2 = CausalBlock1D(dim_out, dim_out)

def subsequent_chunk_mask(
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
) -> paddle.Tensor:
    """Create mask for subsequent steps (size, size) with chunk size,
       this is for streaming encoder

    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks

    Returns:
        paddle.Tensor: mask

    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    pos_idx = paddle.arange(size, dtype='int64')
    block_value = (paddle.floor_divide(pos_idx, chunk_size) + 1) * chunk_size
    ret = pos_idx.unsqueeze(0) < block_value.unsqueeze(1)
    
    return ret


def add_optional_chunk_mask(xs: paddle.Tensor,
                            masks: paddle.Tensor,
                            use_dynamic_chunk: bool,
                            use_dynamic_left_chunk: bool,
                            decoding_chunk_size: int,
                            static_chunk_size: int,
                            num_decoding_left_chunks: int,
                            enable_full_context: bool = True):
    """ Apply optional mask for encoder.

    Args:
        xs (paddle.Tensor): padded input, (B, L, D), L for max length
        mask (paddle.Tensor): mask for xs, (B, 1, L)
        use_dynamic_chunk (bool): whether to use dynamic chunk or not
        use_dynamic_left_chunk (bool): whether to use dynamic left chunk for
            training.
        decoding_chunk_size (int): decoding chunk size for dynamic chunk, it's
            0: default for training, use random dynamic chunk.
            <0: for decoding, use full chunk.
            >0: for decoding, use fixed chunk size as set.
        static_chunk_size (int): chunk size for static chunk training/decoding
            if it's greater than 0, if use_dynamic_chunk is true,
            this parameter will be ignored
        num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
            >=0: use num_decoding_left_chunks
            <0: use all left chunks
        enable_full_context (bool):
            True: chunk size is either [1, 25] or full context(max_len)
            False: chunk size ~ U[1, 25]

    Returns:
        paddle.Tensor: chunk mask of the input xs.
    """
    # Whether to use chunk mask or not
    if use_dynamic_chunk:
        max_len = xs.shape[1]
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            # chunk size is either [1, 25] or full context(max_len).
            # Since we use 4 times subsampling and allow up to 1s(100 frames)
            # delay, the maximum frame is 100 / 4 = 25.
            chunk_size = paddle.randint(1, max_len, shape=(1,)).item()
            num_left_chunks = -1
            if chunk_size > max_len // 2 and enable_full_context:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_left_chunks = paddle.randint(0, max_left_chunks, shape=(1,)).item()
        
        chunk_masks = subsequent_chunk_mask(xs.shape[1], chunk_size,
                                            num_left_chunks)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(xs.shape[1], static_chunk_size,
                                            num_left_chunks)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    else:
        chunk_masks = masks
    assert chunk_masks.dtype == paddle.bool
    if (chunk_masks.sum(axis=-1) == 0).sum().item() != 0:
        print('get chunk_masks all false at some timestep, force set to true, make sure they are masked in future computation!')
        all_false_mask = chunk_masks.sum(axis=-1) == 0
        chunk_masks = paddle.where(all_false_mask.unsqueeze(-1), paddle.ones_like(chunk_masks, dtype='bool'), chunk_masks)
    
    return chunk_masks

def mask_to_bias(mask: paddle.Tensor, dtype: str) -> paddle.Tensor:
    assert mask.dtype == paddle.bool, "Input mask must be of boolean type"
    assert dtype in [paddle.float32, paddle.bfloat16, paddle.float16], f"Unsupported dtype: {dtype}"
    mask = mask.astype(dtype)
    mask = (1.0 - mask) * -1.0e+10
    
    return mask

class ConditionalDecoder(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        channels=(256, 256),
        dropout=0.05,
        attention_head_dim=64,
        n_blocks=1,
        num_mid_blocks=2,
        num_heads=4,
        act_fn="snake",
    ):
        """
        This decoder requires an input with the same shape of the target. So, if your text content
        is shorter or longer than the outputs, please re-sampling it before feeding to the decoder.
        """
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )
        self.down_blocks = nn.LayerList([])
        self.mid_blocks = nn.LayerList([])
        self.up_blocks = nn.LayerList([])

        output_channel = in_channels
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.LayerList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            downsample = (
                Downsample1D(output_channel) if not is_last else nn.Conv1D(output_channel, output_channel, 3, padding=1)
            )
            self.down_blocks.append(nn.LayerList([resnet, transformer_blocks, downsample]))

        for _ in range(num_mid_blocks):
            input_channel = channels[-1]
            out_channels = channels[-1]
            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)

            transformer_blocks = nn.LayerList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )

            self.mid_blocks.append(nn.LayerList([resnet, transformer_blocks]))

        channels = channels[::-1] + (channels[0],)
        for i in range(len(channels) - 1):
            input_channel = channels[i] * 2
            output_channel = channels[i + 1]
            is_last = i == len(channels) - 2
            resnet = ResnetBlock1D(
                dim=input_channel,
                dim_out=output_channel,
                time_emb_dim=time_embed_dim,
            )
            transformer_blocks = nn.LayerList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            upsample = (
                Upsample1D(output_channel, use_conv_transpose=True)
                if not is_last
                else nn.Conv1D(output_channel, output_channel, 3, padding=1)
            )
            self.up_blocks.append(nn.LayerList([resnet, transformer_blocks, upsample]))
        self.final_block = Block1D(channels[-1], channels[-1])
        self.final_proj = nn.Conv1D(channels[-1], self.out_channels, 1)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv1D):
                nn.initializer.KaimingNormal(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.initializer.Constant(m.bias, value=0)
            elif isinstance(m, nn.GroupNorm):
                nn.initializer.Constant(m.weight, value=1)
                nn.initializer.Constant(m.bias, value=0)
            elif isinstance(m, nn.Linear):
                nn.initializer.KaimingNormal(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.initializer.Constant(m.bias, value=0)

    def forward(self, x, mask, mu, t, spks=None, cond=None, streaming=False):
        """Forward pass of the UNet1DConditional model.

        Args:
            x (paddle.Tensor): shape (batch_size, in_channels, time)
            mask (paddle.Tensor): shape (batch_size, 1, time)
            t (paddle.Tensor): shape (batch_size)
            spks (paddle.Tensor, optional): shape: (batch_size, condition_channels). Defaults to None.
            cond (paddle.Tensor, optional): placeholder for future use. Defaults to None.

        Returns:
            paddle.Tensor: output tensor
        """

        t = self.time_embeddings(t).astype(t.dtype)
        t = self.time_mlp(t)

        x = pack([x, mu], "b * t")[0]
        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]
        if cond is not None:
            x = pack([x, cond], "b * t")[0]

        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = add_optional_chunk_mask(x, mask_down.astype('bool'), False, False, 0, 0, -1).repeat(1, x.shape[1], 1)
            attn_mask = mask_to_bias(attn_mask, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t").contiguous()
            hiddens.append(x)  # Save hidden states for skip connections
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = add_optional_chunk_mask(x, mask_mid.astype('bool'), False, False, 0, 0, -1).repeat(1, x.shape[1], 1)
            attn_mask = mask_to_bias(attn_mask, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t").contiguous()

        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = pack([x[:, :, :skip.shape[-1]], skip], "b * t")[0]
            x = resnet(x, mask_up, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = add_optional_chunk_mask(x, mask_up.astype('bool'), False, False, 0, 0, -1).repeat(1, x.shape[1], 1)
            attn_mask = mask_to_bias(attn_mask, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t").contiguous()
            x = upsample(x * mask_up)
        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        return output * mask


class CausalConditionalDecoder(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        channels=(256, 256),
        dropout=0.05,
        attention_head_dim=64,
        n_blocks=1,
        num_mid_blocks=2,
        num_heads=4,
        act_fn="snake",
        static_chunk_size=50,
        num_decoding_left_chunks=2,
    ):
        """
        This decoder requires an input with the same shape of the target. So, if your text content
        is shorter or longer than the outputs, please re-sampling it before feeding to the decoder.
        """
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embeddings = SinusoidalPosEmb(in_channels) 
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding( 
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )
        self.static_chunk_size = static_chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks
        self.down_blocks = nn.LayerList([])
        self.mid_blocks = nn.LayerList([])
        self.up_blocks = nn.LayerList([])

        output_channel = in_channels
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            resnet = CausalResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.LayerList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            downsample = (
                Downsample1D(output_channel) if not is_last else CausalConv1d(output_channel, output_channel, 3)  # 假设已实现
            )
            self.down_blocks.append(nn.LayerList([resnet, transformer_blocks, downsample]))

        for _ in range(num_mid_blocks):
            input_channel = channels[-1]
            out_channels = channels[-1]
            resnet = CausalResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)

            transformer_blocks = nn.LayerList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )

            self.mid_blocks.append(nn.LayerList([resnet, transformer_blocks]))

        channels = channels[::-1] + (channels[0],)
        for i in range(len(channels) - 1):
            input_channel = channels[i] * 2
            output_channel = channels[i + 1]
            is_last = i == len(channels) - 2
            resnet = CausalResnetBlock1D(
                dim=input_channel,
                dim_out=output_channel,
                time_emb_dim=time_embed_dim,
            )
            transformer_blocks = nn.LayerList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            upsample = (
                Upsample1D(output_channel, use_conv_transpose=True)  # 假设已实现
                if not is_last
                else CausalConv1d(output_channel, output_channel, 3)
            )
            self.up_blocks.append(nn.LayerList([resnet, transformer_blocks, upsample]))
        self.final_block = CausalBlock1D(channels[-1], channels[-1])  # 假设已实现
        self.final_proj = nn.Conv1D(channels[-1], self.out_channels, 1)  # 使用 Conv1D
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.sublayers():  # 使用 sublayers() 而不是 modules()
            if isinstance(m, nn.Conv1D):
                nn.initializer.KaimingNormal(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    initializer = nn.initializer.Constant(value=0)
                    initializer(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.initializer.Constant(m.weight, value=1)
                nn.initializer.Constant(m.bias, value=0)
            elif isinstance(m, nn.Linear):
                nn.initializer.KaimingNormal(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    initializer = nn.initializer.Constant(value=0)
                    initializer(m.bias)
    def forward(self, x, mask, mu, t, spks=None, cond=None, streaming=False):
        """Forward pass of the UNet1DConditional model.

        Args:
            x (paddle.Tensor): shape (batch_size, in_channels, time)
            mask (paddle.Tensor): shape (batch_size, 1, time)
            mu (paddle.Tensor): mean tensor for conditioning
            t (paddle.Tensor): shape (batch_size)
            spks (paddle.Tensor, optional): shape: (batch_size, condition_channels). Defaults to None.
            cond (paddle.Tensor, optional): placeholder for future use. Defaults to None.
            streaming (bool, optional): whether to use streaming mode. Defaults to False.

        Returns:
            paddle.Tensor: output tensor
        """
        t = self.time_embeddings(t).astype(t.dtype)  # 使用 astype 代替 .to(t.dtype)
        t = self.time_mlp(t)
        x = pack([x, mu], "b * t")[0]  # 假设 pack 函数已实现
        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])  # 假设 repeat 函数已实现
            x = pack([x, spks], "b * t")[0]
        if cond is not None:
            x = pack([x, cond], "b * t")[0]

        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            
            x = rearrange(x, "b c t -> b t c").contiguous()  # 假设 rearrange 函数已实现
            if streaming is True:
                attn_mask = add_optional_chunk_mask(x, mask_down.astype('bool'), False, False, 0, self.static_chunk_size, -1)  # 使用 astype('bool')
            else:
                attn_mask = add_optional_chunk_mask(x, mask_down.astype('bool'), False, False, 0, 0, -1).repeat(1, x.shape[1], 1)  # 使用 .shape 而不是 .size()
            attn_mask = mask_to_bias(attn_mask, x.dtype)  # 假设 mask_to_bias 函数已实现
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )

            x = rearrange(x, "b t c -> b c t").contiguous()
            hiddens.append(x)  # Save hidden states for skip connections
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            if streaming is True:
                attn_mask = add_optional_chunk_mask(x, mask_mid.astype('bool'), False, False, 0, self.static_chunk_size, -1)
            else:
                attn_mask = add_optional_chunk_mask(x, mask_mid.astype('bool'), False, False, 0, 0, -1).repeat(1, x.shape[1], 1)
            attn_mask = mask_to_bias(attn_mask, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t").contiguous()

        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = pack([x[:, :, :skip.shape[-1]], skip], "b * t")[0]
            x = resnet(x, mask_up, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            if streaming is True:
                attn_mask = add_optional_chunk_mask(x, mask_up.astype('bool'), False, False, 0, self.static_chunk_size, -1)
            else:
                attn_mask = add_optional_chunk_mask(x, mask_up.astype('bool'), False, False, 0, 0, -1).repeat(1, x.shape[1], 1)
            attn_mask = mask_to_bias(attn_mask, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t").contiguous()
            x = upsample(x * mask_up)
        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        return output * mask
