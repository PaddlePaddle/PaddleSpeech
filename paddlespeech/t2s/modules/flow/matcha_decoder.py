import math
from typing import Optional

import einops
import paddle
from conformer import ConformerBlock
from matcha.models.components.transformer import BasicTransformerBlock
ACTIVATION_FUNCTIONS = {
    "swish": paddle.nn.SiLU(),
    "silu": paddle.nn.SiLU(),
    "mish": paddle.nn.Mish(),
    "gelu": paddle.nn.GELU(),
    "relu": paddle.nn.ReLU(),
}
def get_activation(act_fn: str) -> paddle.nn.Layer:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """
    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")

class SinusoidalPosEmb(paddle.nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.place
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = paddle.exp(x=paddle.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = paddle.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block1D(paddle.nn.Layer):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = paddle.nn.Sequential(
            paddle.nn.Conv1d(dim, dim_out, 3, padding=1),
            paddle.nn.GroupNorm(num_groups=groups, num_channels=dim_out),
            paddle.nn.Mish(),
        )

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock1D(paddle.nn.Layer):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = paddle.nn.Sequential(
            paddle.nn.Mish(),
            paddle.nn.Linear(in_features=time_emb_dim, out_features=dim_out),
        )
        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)
        self.res_conv = paddle.nn.Conv1d(dim, dim_out, 1)

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class Downsample1D(paddle.nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.conv = paddle.nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class TimestepEmbedding(paddle.nn.Layer):
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
        self.linear_1 = paddle.nn.Linear(
            in_features=in_channels, out_features=time_embed_dim
        )
        if cond_proj_dim is not None:
            self.cond_proj = paddle.nn.Linear(
                in_features=cond_proj_dim, out_features=in_channels, bias_attr=False
            )
        else:
            self.cond_proj = None
        self.act = get_activation(act_fn)
        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = paddle.nn.Linear(
            in_features=time_embed_dim, out_features=time_embed_dim_out
        )
        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)
        if self.act is not None:
            sample = self.act(sample)
        sample = self.linear_2(sample)
        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Upsample1D(paddle.nn.Layer):
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

    def __init__(
        self,
        channels,
        use_conv=False,
        use_conv_transpose=True,
        out_channels=None,
        name="conv",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.conv = None
        if use_conv_transpose:
            self.conv = paddle.nn.Conv1DTranspose(
                in_channels=channels,
                out_channels=self.out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        elif use_conv:
            self.conv = paddle.nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, inputs):
        assert inputs.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(inputs)
        outputs = paddle.nn.functional.interpolate(
            x=inputs, scale_factor=2.0, mode="nearest"
        )
        if self.use_conv:
            outputs = self.conv(outputs)
        return outputs


class ConformerWrapper(ConformerBlock):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0,
        ff_dropout=0,
        conv_dropout=0,
        conv_causal=False,
    ):
        super().__init__(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            conv_dropout=conv_dropout,
            conv_causal=conv_causal,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
    ):
        return super().forward(x=hidden_states, mask=attention_mask.bool())


class Decoder(paddle.nn.Layer):
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
        down_block_type="transformer",
        mid_block_type="transformer",
        up_block_type="transformer",
    ):
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels, time_embed_dim=time_embed_dim, act_fn="silu"
        )
        self.down_blocks = paddle.nn.LayerList(sublayers=[])
        self.mid_blocks = paddle.nn.LayerList(sublayers=[])
        self.up_blocks = paddle.nn.LayerList(sublayers=[])
        output_channel = in_channels
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            resnet = ResnetBlock1D(
                dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim
            )
            transformer_blocks = paddle.nn.LayerList(
                sublayers=[
                    self.get_block(
                        down_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            downsample = (
                Downsample1D(output_channel)
                if not is_last
                else paddle.nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )
            self.down_blocks.append(
                paddle.nn.LayerList(sublayers=[resnet, transformer_blocks, downsample])
            )
        for i in range(num_mid_blocks):
            input_channel = channels[-1]
            out_channels = channels[-1]
            resnet = ResnetBlock1D(
                dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim
            )
            transformer_blocks = paddle.nn.LayerList(
                sublayers=[
                    self.get_block(
                        mid_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            self.mid_blocks.append(
                paddle.nn.LayerList(sublayers=[resnet, transformer_blocks])
            )
        channels = channels[::-1] + (channels[0],)
        for i in range(len(channels) - 1):
            input_channel = channels[i]
            output_channel = channels[i + 1]
            is_last = i == len(channels) - 2
            resnet = ResnetBlock1D(
                dim=2 * input_channel,
                dim_out=output_channel,
                time_emb_dim=time_embed_dim,
            )
            transformer_blocks = paddle.nn.LayerList(
                sublayers=[
                    self.get_block(
                        up_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            upsample = (
                Upsample1D(output_channel, use_conv_transpose=True)
                if not is_last
                else paddle.nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )
            self.up_blocks.append(
                paddle.nn.LayerList(sublayers=[resnet, transformer_blocks, upsample])
            )
        self.final_block = Block1D(channels[-1], channels[-1])
        self.final_proj = paddle.nn.Conv1d(channels[-1], self.out_channels, 1)
        self.initialize_weights()

    @staticmethod
    def get_block(block_type, dim, attention_head_dim, num_heads, dropout, act_fn):
        if block_type == "conformer":
            block = ConformerWrapper(
                dim=dim,
                dim_head=attention_head_dim,
                heads=num_heads,
                ff_mult=1,
                conv_expansion_factor=2,
                ff_dropout=dropout,
                attn_dropout=dropout,
                conv_dropout=dropout,
                conv_kernel_size=31,
            )
        elif block_type == "transformer":
            block = BasicTransformerBlock(
                dim=dim,
                num_attention_heads=num_heads,
                attention_head_dim=attention_head_dim,
                dropout=dropout,
                activation_fn=act_fn,
            )
        else:
            raise ValueError(f"Unknown block type {block_type}")
        return block

    def initialize_weights(self):
        for m in self.sublayers():
            if isinstance(m, paddle.nn.Conv1d):
                paddle.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    paddle.nn.init.constant_(m.bias, 0)
            elif isinstance(m, paddle.nn.GroupNorm):
                paddle.nn.init.constant_(m.weight, 1)
                paddle.nn.init.constant_(m.bias, 0)
            elif isinstance(m, paddle.nn.Linear):
                paddle.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    paddle.nn.init.constant_(m.bias, 0)

    def forward(self, x, mask, mu, t, spks=None, cond=None):
        """Forward pass of the UNet1DConditional model.

        Args:
            x (torch.Tensor): shape (batch_size, in_channels, time)
            mask (_type_): shape (batch_size, 1, time)
            t (_type_): shape (batch_size)
            spks (_type_, optional): shape: (batch_size, condition_channels). Defaults to None.
            cond (_type_, optional): placeholder for future use. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        t = self.time_embeddings(t)
        t = self.time_mlp(t)
        x = einops.pack([x, mu], "b * t")[0]
        if spks is not None:
            spks = einops.repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = einops.pack([x, spks], "b * t")[0]
        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            x = einops.rearrange(x, "b c t -> b t c")
            mask_down = einops.rearrange(mask_down, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x, attention_mask=mask_down, timestep=t
                )
            x = einops.rearrange(x, "b t c -> b c t")
            mask_down = einops.rearrange(mask_down, "b t -> b 1 t")
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]
        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = einops.rearrange(x, "b c t -> b t c")
            mask_mid = einops.rearrange(mask_mid, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x, attention_mask=mask_mid, timestep=t
                )
            x = einops.rearrange(x, "b t c -> b c t")
            mask_mid = einops.rearrange(mask_mid, "b t -> b 1 t")
        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            x = resnet(einops.pack([x, hiddens.pop()], "b * t")[0], mask_up, t)
            x = einops.rearrange(x, "b c t -> b t c")
            mask_up = einops.rearrange(mask_up, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x, attention_mask=mask_up, timestep=t
                )
            x = einops.rearrange(x, "b t c -> b c t")
            mask_up = einops.rearrange(mask_up, "b t -> b 1 t")
            x = upsample(x * mask_up)
        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        return output * mask
