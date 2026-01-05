import paddle

"""Encoder definition."""
from typing import Tuple
import paddle.nn.functional as F
from paddlespeech.t2s.modules.transformer.convolution import ConvolutionModule
from paddlespeech.t2s.modules.transformer.encoder_layer import ConformerEncoderLayer
from paddlespeech.t2s.modules.transformer.positionwise_feed_forward import PositionwiseFeedForward
from paddlespeech.t2s.models.CosyVoice.class_utils import (COSYVOICE_ACTIVATION_CLASSES,
                                         COSYVOICE_ATTENTION_CLASSES,
                                         COSYVOICE_EMB_CLASSES,
                                         COSYVOICE_SUBSAMPLE_CLASSES)
from paddlespeech.t2s.modules.transformer.mask import add_optional_chunk_mask, make_pad_mask


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

    def __init__(self, channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv = paddle.nn.Conv1D(
            self.channels, self.out_channels, stride * 2 + 1, stride=1, padding=0
        )

    def forward(
        self, inputs: paddle.Tensor, input_lengths: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        inputs = inputs.unsqueeze(2)
        outputs = paddle.nn.functional.interpolate(
            x=inputs, scale_factor=[1, float(self.stride)],mode="nearest"
        )
        outputs = outputs.squeeze(2)
        outputs = F.pad(outputs, [self.stride * 2, 0], value=0.0)
        outputs = self.conv(outputs)
        return outputs, input_lengths * self.stride


class PreLookaheadLayer(paddle.nn.Layer):
    def __init__(self, channels: int, pre_lookahead_len: int = 1):
        super().__init__()
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = paddle.nn.Conv1D(
            channels, channels, kernel_size=pre_lookahead_len + 1, stride=1, padding=0
        )
        self.conv2 = paddle.nn.Conv1D(
            channels, channels, kernel_size=3, stride=1, padding=0
        )

    def forward(
        self, inputs: paddle.Tensor, context: paddle.Tensor = paddle.zeros([0, 0, 0])
    ) -> paddle.Tensor:
        """
        inputs: (batch_size, seq_len, channels)
        """
        outputs =  paddle.transpose(inputs, perm=[0, 2, 1]).contiguous()
        context = paddle.transpose(context, perm=[0, 2, 1]).contiguous()
        if context.shape[2] == 0:
            outputs = F.pad(
                outputs, [0, self.pre_lookahead_len], mode="constant", value=0.0
            )
        else:
            assert (
                self.training is False
            ), "you have passed context, make sure that you are running inference mode"
            assert context.shape[2] == self.pre_lookahead_len
            outputs = F.pad(
                paddle.cat([outputs, context], dim=2),
                [0, self.pre_lookahead_len - context.shape[2]],
                mode="constant",
                value=0.0,
            )
        outputs = paddle.nn.functional.leaky_relu(x=self.conv1(outputs))
        outputs = F.pad(
            outputs, [self.conv2._kernel_size[0] - 1, 0], mode="constant", value=0.0
        )
        outputs = self.conv2(outputs)
        outputs = paddle.transpose(outputs, perm=[0, 2, 1]).contiguous()

        outputs = outputs + inputs
        return outputs


class UpsampleConformerEncoder(paddle.nn.Layer):
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: paddle.nn.Layer = None,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = True,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        key_bias: bool = True,
        gradient_checkpointing: bool = False,
    ):
        """
        Args:
            input_size (int): input dim
            output_size (int): dimension of attention
            attention_heads (int): the number of heads of multi head attention
            linear_units (int): the hidden units number of position-wise feed
                forward
            num_blocks (int): the number of decoder blocks
            dropout_rate (float): dropout rate
            attention_dropout_rate (float): dropout rate in attention
            positional_dropout_rate (float): dropout rate after adding
                positional encoding
            input_layer (str): input layer type.
                optional [linear, conv2d, conv2d6, conv2d8]
            pos_enc_layer_type (str): Encoder positional encoding layer type.
                opitonal [abs_pos, scaled_abs_pos, rel_pos, no_pos]
            normalize_before (bool):
                True: use layer_norm before each sub-block of a layer.
                False: use layer_norm after each sub-block of a layer.
            static_chunk_size (int): chunk size for static chunk training and
                decoding
            use_dynamic_chunk (bool): whether use dynamic chunk size for
                training or not, You can only use fixed chunk(chunk_size > 0)
                or dyanmic chunk size(use_dynamic_chunk = True)
            global_cmvn (Optional[torch.nn.Module]): Optional GlobalCMVN module
            use_dynamic_left_chunk (bool): whether use dynamic left chunk in
                dynamic chunk training
            key_bias: whether use bias in attention.linear_k, False for whisper models.
            gradient_checkpointing: rerunning a forward-pass segment for each
                checkpointed segment during backward.
        """
        super().__init__()
        self._output_size = output_size
        self.global_cmvn = global_cmvn
        self.embed = COSYVOICE_SUBSAMPLE_CLASSES[input_layer](
            input_size,
            output_size,
            dropout_rate,
            COSYVOICE_EMB_CLASSES[pos_enc_layer_type](
                output_size, positional_dropout_rate
            ),
        )
        self.normalize_before = normalize_before
        self.after_norm = paddle.nn.LayerNorm(
            normalized_shape=output_size, epsilon=1e-05
        )
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.gradient_checkpointing = gradient_checkpointing
        activation = COSYVOICE_ACTIVATION_CLASSES[activation_type]()
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            False,
        )
        positionwise_layer_args = (output_size, linear_units, dropout_rate, activation)
        convolution_layer_args = (
            output_size,
            cnn_module_kernel,
            activation,
            cnn_module_norm,
            causal,
        )
        self.pre_lookahead_layer = PreLookaheadLayer(channels=512, pre_lookahead_len=3)
        self.encoders = paddle.nn.LayerList(
            sublayers=[
                ConformerEncoderLayer(
                    output_size,
                    COSYVOICE_ATTENTION_CLASSES[selfattention_layer_type](
                        *encoder_selfattn_layer_args
                    ),
                    PositionwiseFeedForward(*positionwise_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args)
                    if macaron_style
                    else None,
                    ConvolutionModule(*convolution_layer_args)
                    if use_cnn_module
                    else None,
                    dropout_rate,
                    normalize_before,
                )
                for _ in range(num_blocks)
            ]
        )
        self.up_layer = Upsample1D(channels=512, out_channels=512, stride=2)
        self.up_embed = COSYVOICE_SUBSAMPLE_CLASSES[input_layer](
            input_size,
            output_size,
            dropout_rate,
            COSYVOICE_EMB_CLASSES[pos_enc_layer_type](
                output_size, positional_dropout_rate
            ),
        )
        self.up_encoders = paddle.nn.LayerList(
            sublayers=[
                ConformerEncoderLayer(
                    output_size,
                    COSYVOICE_ATTENTION_CLASSES[selfattention_layer_type](
                        *encoder_selfattn_layer_args
                    ),
                    PositionwiseFeedForward(*positionwise_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args)
                    if macaron_style
                    else None,
                    ConvolutionModule(*convolution_layer_args)
                    if use_cnn_module
                    else None,
                    dropout_rate,
                    normalize_before,
                )
                for _ in range(4)
            ]
        )

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs: paddle.Tensor,
        xs_lens: paddle.Tensor,
        context: paddle.Tensor = paddle.zeros([0, 0, 0]),
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
        streaming: bool = False,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        NOTE(xcsong):
            We pass the `__call__` method of the modules instead of `forward` to the
            checkpointing API because `__call__` attaches all the hooks of the module.
            https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2
        """
        T = xs.shape[1]
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        if context.shape[1] != 0:
            assert (
                self.training is False
            ), "you have passed context, make sure that you are running inference mode"
            context_masks = paddle.ones(1, 1, context.shape[1]).to(masks)
            context, _, _ = self.embed(context, context_masks, offset=xs.shape[1])
        mask_pad = masks
        chunk_masks = add_optional_chunk_mask(
            xs,
            masks,
            False,
            False,
            0,
            self.static_chunk_size if streaming is True else 0,
            -1,
        )
        xs = self.pre_lookahead_layer(xs, context=context)
        xs = self.forward_layers(xs, chunk_masks, pos_emb, mask_pad)
        # 
        xs = paddle.transpose(xs, perm=[0, 2, 1]).contiguous()
        xs, xs_lens = self.up_layer(xs, xs_lens)
        xs = paddle.transpose(xs, perm=[0, 2, 1]).contiguous()
        T = xs.shape[1]
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)
        xs, pos_emb, masks = self.up_embed(xs, masks)
        mask_pad = masks
        chunk_masks = add_optional_chunk_mask(
            xs,
            masks,
            False,
            False,
            0,
            self.static_chunk_size * self.up_layer.stride if streaming is True else 0,
            -1,
        )
        xs = self.forward_up_layers(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks

    def forward_layers(
        self,
        xs: paddle.Tensor,
        chunk_masks: paddle.Tensor,
        pos_emb: paddle.Tensor,
        mask_pad: paddle.Tensor,
    ) -> paddle.Tensor:
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs

    def forward_up_layers(
        self,
        xs: paddle.Tensor,
        chunk_masks: paddle.Tensor,
        pos_emb: paddle.Tensor,
        mask_pad: paddle.Tensor,
    ) -> paddle.Tensor:
        for layer in self.up_encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs
