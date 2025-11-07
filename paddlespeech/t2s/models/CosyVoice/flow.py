import math
from typing import Any
from typing import Dict
from typing import List

import paddle
from paddle import nn
from paddle.nn import functional as F

class Decoder(nn.Layer):
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
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )

        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        output_channel = in_channels
        for i in range(len(channels)):  # pylint: disable=consider-using-enumerate
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList(
                [
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
                Downsample1D(output_channel) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )

            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))

        for i in range(num_mid_blocks):
            input_channel = channels[-1]
            out_channels = channels[-1]

            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)

            transformer_blocks = nn.ModuleList(
                [
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

            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

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
            transformer_blocks = nn.ModuleList(
                [
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
                else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )

            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))

        self.final_block = Block1D(channels[-1], channels[-1])
        self.final_proj = nn.Conv1d(channels[-1], self.out_channels, 1)

        self.initialize_weights()
        # nn.init.normal_(self.final_proj.weight)

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
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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

        x = pack([x, mu], "b * t")[0]

        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]

        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            x = rearrange(x, "b c t -> b t c")
            mask_down = rearrange(mask_down, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_down,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_down = rearrange(mask_down, "b t -> b 1 t")
            hiddens.append(x)  # Save hidden states for skip connections
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = rearrange(x, "b c t -> b t c")
            mask_mid = rearrange(mask_mid, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_mid,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_mid = rearrange(mask_mid, "b t -> b 1 t")

        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            x = resnet(pack([x, hiddens.pop()], "b * t")[0], mask_up, t)
            x = rearrange(x, "b c t -> b t c")
            mask_up = rearrange(mask_up, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_up,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_up = rearrange(mask_up, "b t -> b 1 t")
            x = upsample(x * mask_up)

        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)

        return output * mask
