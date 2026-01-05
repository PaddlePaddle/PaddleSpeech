import paddle

"""Encoder self-attention layer definition."""
from typing import Optional, Tuple


class TransformerEncoderLayer(paddle.nn.Layer):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
    """

    def __init__(
        self,
        size: int,
        self_attn: paddle.nn.Layer,
        feed_forward: paddle.nn.Layer,
        dropout_rate: float,
        normalize_before: bool = True,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = paddle.nn.LayerNorm(normalized_shape=size, epsilon=1e-12)
        self.norm2 = paddle.nn.LayerNorm(normalized_shape=size, epsilon=1e-12)
        self.dropout = paddle.nn.Dropout(p=dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

    def forward(
        self,
        x: paddle.Tensor,
        mask: paddle.Tensor,
        pos_emb: paddle.Tensor,
        mask_pad: paddle.Tensor = paddle.ones((0, 0, 0), dtype=paddle.bool),
        att_cache: paddle.Tensor = paddle.zeros((0, 0, 0, 0)),
        cnn_cache: paddle.Tensor = paddle.zeros((0, 0, 0, 0)),
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time，time),
                (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): just for interface compatibility
                to ConformerEncoderLayer
            mask_pad (torch.Tensor): does not used in transformer layer,
                just for unified api with conformer.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
                (#batch=1, size, cache_t2), not used here, it's for interface
                compatibility to ConformerEncoderLayer.
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time).
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cahce tensor (#batch=1, size, cache_t2).

        """
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        x_att, new_att_cache = self.self_attn(
            x, x, x, mask, pos_emb=pos_emb, cache=att_cache
        )
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm1(x)
        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)
        fake_cnn_cache = paddle.zeros((0, 0, 0), dtype=x.dtype, device=x.place)
        return x, mask, new_att_cache, fake_cnn_cache


class ConformerEncoderLayer(paddle.nn.Layer):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module
             instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: use layer_norm after each sub-block.
    """

    def __init__(
        self,
        size: int,
        self_attn: paddle.nn.Layer,
        feed_forward: Optional[paddle.nn.Layer] = None,
        feed_forward_macaron: Optional[paddle.nn.Layer] = None,
        conv_module: Optional[paddle.nn.Layer] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = paddle.nn.LayerNorm(normalized_shape=size, epsilon=1e-12)
        self.norm_mha = paddle.nn.LayerNorm(normalized_shape=size, epsilon=1e-12)
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = paddle.nn.LayerNorm(
                normalized_shape=size, epsilon=1e-12
            )
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = paddle.nn.LayerNorm(normalized_shape=size, epsilon=1e-12)
            self.norm_final = paddle.nn.LayerNorm(normalized_shape=size, epsilon=1e-12)
        self.dropout = paddle.nn.Dropout(p=dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

    def forward(
        self,
        x: paddle.Tensor,
        mask: paddle.Tensor,
        pos_emb: paddle.Tensor,
        mask_pad: paddle.Tensor = paddle.ones((0, 0, 0), dtype=paddle.bool),
        att_cache: paddle.Tensor = paddle.zeros((0, 0, 0, 0)),
        cnn_cache: paddle.Tensor = paddle.zeros((0, 0, 0, 0)),
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time，time),
                (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): positional encoding, must not be None
                for ConformerEncoderLayer.
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1，time), (0, 0, 0) means fake mask.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
                (#batch=1, size, cache_t2)
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time).
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cahce tensor (#batch, size, cache_t2).
        """
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb, att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)
        new_cnn_cache = paddle.zeros((0, 0, 0), dtype=x.dtype, device=x.place)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)
            if not self.normalize_before:
                x = self.norm_conv(x)
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)
        if self.conv_module is not None:
            x = self.norm_final(x)
        return x, mask, new_att_cache, new_cnn_cache
