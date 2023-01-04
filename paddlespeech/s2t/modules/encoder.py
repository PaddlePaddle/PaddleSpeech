# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2019 Mobvoi Inc. All Rights Reserved.
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
# Modified from wenet(https://github.com/wenet-e2e/wenet)
"""Encoder definition."""
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import paddle
from paddle import nn
from typeguard import check_argument_types

from paddlespeech.s2t.modules.activation import get_activation
from paddlespeech.s2t.modules.align import LayerNorm
from paddlespeech.s2t.modules.align import Linear
from paddlespeech.s2t.modules.attention import MultiHeadedAttention
from paddlespeech.s2t.modules.attention import RelPositionMultiHeadedAttention
from paddlespeech.s2t.modules.attention import RelPositionMultiHeadedAttention2
from paddlespeech.s2t.modules.conformer_convolution import ConvolutionModule
from paddlespeech.s2t.modules.embedding import NoPositionalEncoding
from paddlespeech.s2t.modules.embedding import PositionalEncoding
from paddlespeech.s2t.modules.embedding import RelPositionalEncoding
from paddlespeech.s2t.modules.encoder_layer import ConformerEncoderLayer
from paddlespeech.s2t.modules.encoder_layer import SqueezeformerEncoderLayer
from paddlespeech.s2t.modules.encoder_layer import TransformerEncoderLayer
from paddlespeech.s2t.modules.mask import add_optional_chunk_mask
from paddlespeech.s2t.modules.mask import make_non_pad_mask
from paddlespeech.s2t.modules.positionwise_feed_forward import PositionwiseFeedForward
from paddlespeech.s2t.modules.subsampling import Conv2dSubsampling4
from paddlespeech.s2t.modules.subsampling import Conv2dSubsampling6
from paddlespeech.s2t.modules.subsampling import Conv2dSubsampling8
from paddlespeech.s2t.modules.subsampling import DepthwiseConv2DSubsampling4
from paddlespeech.s2t.modules.subsampling import LinearNoSubsampling
from paddlespeech.s2t.modules.subsampling import TimeReductionLayer1D
from paddlespeech.s2t.modules.subsampling import TimeReductionLayer2D
from paddlespeech.s2t.modules.subsampling import TimeReductionLayerStream
from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()

__all__ = [
    "BaseEncoder", 'TransformerEncoder', "ConformerEncoder",
    "SqueezeformerEncoder"
]


class BaseEncoder(nn.Layer):
    def __init__(self,
                 input_size: int,
                 output_size: int=256,
                 attention_heads: int=4,
                 linear_units: int=2048,
                 num_blocks: int=6,
                 dropout_rate: float=0.1,
                 positional_dropout_rate: float=0.1,
                 attention_dropout_rate: float=0.0,
                 input_layer: str="conv2d",
                 pos_enc_layer_type: str="abs_pos",
                 normalize_before: bool=True,
                 concat_after: bool=False,
                 static_chunk_size: int=0,
                 use_dynamic_chunk: bool=False,
                 global_cmvn: paddle.nn.Layer=None,
                 use_dynamic_left_chunk: bool=False,
                 max_len: int=5000):
        """
        Args:
            input_size (int): input dim, d_feature
            output_size (int): dimension of attention, d_model
            attention_heads (int): the number of heads of multi head attention
            linear_units (int): the hidden units number of position-wise feed
                forward
            num_blocks (int): the number of encoder blocks
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
            concat_after (bool): whether to concat attention layer's input
                and output.
                True: x -> x + linear(concat(x, att(x)))
                False: x -> x + att(x)
            static_chunk_size (int): chunk size for static chunk training and
                decoding
            use_dynamic_chunk (bool): whether use dynamic chunk size for
                training or not, You can only use fixed chunk(chunk_size > 0)
                or dyanmic chunk size(use_dynamic_chunk = True)
            global_cmvn (Optional[paddle.nn.Layer]): Optional GlobalCMVN layer
            use_dynamic_left_chunk (bool): whether use dynamic left chunk in
                dynamic chunk training
        """
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "no_pos":
            pos_enc_class = NoPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            subsampling_class = LinearNoSubsampling
        elif input_layer == "conv2d":
            subsampling_class = Conv2dSubsampling4
        elif input_layer == "conv2d6":
            subsampling_class = Conv2dSubsampling6
        elif input_layer == "conv2d8":
            subsampling_class = Conv2dSubsampling8
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.global_cmvn = global_cmvn
        self.embed = subsampling_class(
            idim=input_size,
            odim=output_size,
            dropout_rate=dropout_rate,
            pos_enc_class=pos_enc_class(
                d_model=output_size,
                dropout_rate=positional_dropout_rate,
                max_len=max_len), )

        self.normalize_before = normalize_before
        self.after_norm = LayerNorm(output_size, epsilon=1e-12)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk

    def output_size(self) -> int:
        return self._output_size

    def forward(
            self,
            xs: paddle.Tensor,
            xs_lens: paddle.Tensor,
            decoding_chunk_size: int=0,
            num_decoding_left_chunks: int=-1,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Embed positions in tensor.
        Args:
            xs: padded input tensor (B, L, D)
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
            encoder output tensor, lens and mask
        """
        masks = make_non_pad_mask(xs_lens).unsqueeze(1)  # (B, 1, L)

        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks, offset=0)
        mask_pad = ~masks
        chunk_masks = add_optional_chunk_mask(
            xs, masks, self.use_dynamic_chunk, self.use_dynamic_left_chunk,
            decoding_chunk_size, self.static_chunk_size,
            num_decoding_left_chunks)
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    def forward_chunk(
            self,
            xs: paddle.Tensor,
            offset: int,
            required_cache_size: int,
            att_cache: paddle.Tensor=paddle.zeros([0, 0, 0, 0]),
            cnn_cache: paddle.Tensor=paddle.zeros([0, 0, 0, 0]),
            att_mask: paddle.Tensor=paddle.ones([0, 0, 0], dtype=paddle.bool)
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """ Forward just one chunk
        Args:
            xs (paddle.Tensor): chunk audio feat input, [B=1, T, D], where 
                `T==(chunk_size-1)*subsampling_rate + subsample.right_context + 1`
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            att_cache(paddle.Tensor): cache tensor for key & val in 
                transformer/conformer attention. Shape is 
                (elayers, head, cache_t1, d_k * 2), where`head * d_k == hidden-dim` 
                and `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (paddle.Tensor): cache tensor for cnn_module in conformer, 
                (elayers, B=1, hidden-dim, cache_t2), where `cache_t2 == cnn.lorder - 1`
        Returns:
            paddle.Tensor: output of current input xs, (B=1, chunk_size, hidden-dim)
            paddle.Tensor: new attention cache required for next chunk, dyanmic shape 
                (elayers, head, T, d_k*2) depending on required_cache_size
            paddle.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache
        """
        assert xs.shape[0] == 1  # batch size must be one
        # tmp_masks is just for interface compatibility, [B=1, C=1, T]
        tmp_masks = paddle.ones([1, 1, xs.shape[1]], dtype=paddle.bool)

        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)

        # before embed, xs=(B, T, D1), pos_emb=(B=1, T, D)
        xs, pos_emb, _ = self.embed(xs, tmp_masks, offset=offset)
        # after embed, xs=(B=1, chunk_size, hidden-dim)

        elayers, _, cache_t1, _ = att_cache.shape
        chunk_size = xs.shape[1]
        attention_key_size = cache_t1 + chunk_size

        # only used when using `RelPositionMultiHeadedAttention`
        pos_emb = self.embed.position_encoding(
            offset=offset - cache_t1, size=attention_key_size)

        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = attention_key_size
        else:
            next_cache_start = max(attention_key_size - required_cache_size, 0)

        r_att_cache = []
        r_cnn_cache = []
        for i, layer in enumerate(self.encoders):
            # att_cache[i:i+1] = (1, head, cache_t1, d_k*2)
            # cnn_cache[i:i+1] = (1, B=1, hidden-dim, cache_t2)

            # WARNING: eliminate if-else cond op in graph
            # tensor zeros([0,0,0,0]) support [i:i+1] slice, will return zeros([0,0,0,0]) tensor
            # raw code as below:
            #   att_cache=att_cache[i:i+1] if elayers > 0 else att_cache,
            #   cnn_cache=cnn_cache[i:i+1] if cnn_cache.shape[0] > 0 else cnn_cache,
            xs, _, new_att_cache, new_cnn_cache = layer(
                xs,
                att_mask,
                pos_emb,
                att_cache=att_cache[i:i + 1],
                cnn_cache=cnn_cache[i:i + 1], )
            # new_att_cache = (1, head, attention_key_size, d_k*2)
            # new_cnn_cache = (B=1, hidden-dim, cache_t2)
            r_att_cache.append(new_att_cache[:, :, next_cache_start:, :])
            r_cnn_cache.append(new_cnn_cache)  # add elayer dim

        if self.normalize_before:
            xs = self.after_norm(xs)

        # r_att_cache (elayers, head, T, d_k*2)
        # r_cnn_cache (elayers, B=1, hidden-dim, cache_t2)
        r_att_cache = paddle.concat(r_att_cache, axis=0)
        r_cnn_cache = paddle.stack(r_cnn_cache, axis=0)
        return xs, r_att_cache, r_cnn_cache

    def forward_chunk_by_chunk(
            self,
            xs: paddle.Tensor,
            decoding_chunk_size: int,
            num_decoding_left_chunks: int=-1,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """ Forward input chunk by chunk with chunk_size like a streaming
            fashion
        Here we should pay special attention to computation cache in the
        streaming style forward chunk by chunk. Three things should be taken
        into account for computation in the current network:
            1. transformer/conformer encoder layers output cache
            2. convolution in conformer
            3. convolution in subsampling
        However, we don't implement subsampling cache for:
            1. We can control subsampling module to output the right result by
               overlapping input instead of cache left context, even though it
               wastes some computation, but subsampling only takes a very
               small fraction of computation in the whole model.
            2. Typically, there are several covolution layers with subsampling
               in subsampling module, it is tricky and complicated to do cache
               with different convolution layers with different subsampling
               rate.
            3. Currently, nn.Sequential is used to stack all the convolution
               layers in subsampling, we need to rewrite it to make it work
               with cache, which is not prefered.
        Args:
            xs (paddle.Tensor): (1, max_len, dim)
            chunk_size (int): decoding chunk size.
            num_left_chunks (int): decoding with num left chunks.
        """
        assert decoding_chunk_size > 0
        # The model is trained by static or dynamic chunk
        assert self.static_chunk_size > 0 or self.use_dynamic_chunk

        # feature stride and window for `subsampling` module
        subsampling = self.embed.subsampling_rate
        context = self.embed.right_context + 1  # Add current frame
        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context

        num_frames = xs.shape[1]
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks

        att_cache: paddle.Tensor = paddle.zeros([0, 0, 0, 0])
        cnn_cache: paddle.Tensor = paddle.zeros([0, 0, 0, 0])

        outputs = []
        offset = 0
        # Feed forward overlap input step by step
        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + decoding_window, num_frames)
            chunk_xs = xs[:, cur:end, :]

            (y, att_cache, cnn_cache) = self.forward_chunk(
                chunk_xs, offset, required_cache_size, att_cache, cnn_cache)

            outputs.append(y)
            offset += y.shape[1]
        ys = paddle.cat(outputs, 1)
        masks = paddle.ones([1, 1, ys.shape[1]], dtype=paddle.bool)
        return ys, masks


class TransformerEncoder(BaseEncoder):
    """Transformer encoder module."""

    def __init__(
            self,
            input_size: int,
            output_size: int=256,
            attention_heads: int=4,
            linear_units: int=2048,
            num_blocks: int=6,
            dropout_rate: float=0.1,
            positional_dropout_rate: float=0.1,
            attention_dropout_rate: float=0.0,
            input_layer: str="conv2d",
            pos_enc_layer_type: str="abs_pos",
            normalize_before: bool=True,
            concat_after: bool=False,
            static_chunk_size: int=0,
            use_dynamic_chunk: bool=False,
            global_cmvn: nn.Layer=None,
            use_dynamic_left_chunk: bool=False, ):
        """ Construct TransformerEncoder
        See Encoder for the meaning of each parameter.
        """
        assert check_argument_types()
        super().__init__(input_size, output_size, attention_heads, linear_units,
                         num_blocks, dropout_rate, positional_dropout_rate,
                         attention_dropout_rate, input_layer,
                         pos_enc_layer_type, normalize_before, concat_after,
                         static_chunk_size, use_dynamic_chunk, global_cmvn,
                         use_dynamic_left_chunk)
        self.encoders = nn.LayerList([
            TransformerEncoderLayer(
                size=output_size,
                self_attn=MultiHeadedAttention(attention_heads, output_size,
                                               attention_dropout_rate),
                feed_forward=PositionwiseFeedForward(output_size, linear_units,
                                                     dropout_rate),
                dropout_rate=dropout_rate,
                normalize_before=normalize_before,
                concat_after=concat_after) for _ in range(num_blocks)
        ])

    def forward_one_step(
            self,
            xs: paddle.Tensor,
            masks: paddle.Tensor,
            cache=None, ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Encode input frame.

        Args:
            xs (paddle.Tensor): (Prefix) Input tensor. (B, T, D)
            masks (paddle.Tensor): Mask tensor. (B, T, T)
            cache (List[paddle.Tensor]): List of cache tensors.

        Returns:
            paddle.Tensor: Output tensor.
            paddle.Tensor: Mask tensor.
            List[paddle.Tensor]: List of new cache tensors.
        """
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)

        xs, pos_emb, masks = self.embed(xs, masks, offset=0)
        if cache is None:
            cache = [None for _ in range(len(self.encoders))]
        new_cache = []
        for c, e in zip(cache, self.encoders):
            xs, masks, _ = e(xs, masks, output_cache=c)
            new_cache.append(xs)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks, new_cache


class ConformerEncoder(BaseEncoder):
    """Conformer encoder module."""

    def __init__(self,
                 input_size: int,
                 output_size: int=256,
                 attention_heads: int=4,
                 linear_units: int=2048,
                 num_blocks: int=6,
                 dropout_rate: float=0.1,
                 positional_dropout_rate: float=0.1,
                 attention_dropout_rate: float=0.0,
                 input_layer: str="conv2d",
                 pos_enc_layer_type: str="rel_pos",
                 normalize_before: bool=True,
                 concat_after: bool=False,
                 static_chunk_size: int=0,
                 use_dynamic_chunk: bool=False,
                 global_cmvn: nn.Layer=None,
                 use_dynamic_left_chunk: bool=False,
                 positionwise_conv_kernel_size: int=1,
                 macaron_style: bool=True,
                 selfattention_layer_type: str="rel_selfattn",
                 activation_type: str="swish",
                 use_cnn_module: bool=True,
                 cnn_module_kernel: int=15,
                 causal: bool=False,
                 cnn_module_norm: str="batch_norm",
                 max_len: int=5000):
        """Construct ConformerEncoder
        Args:
            input_size to use_dynamic_chunk, see in BaseEncoder
            positionwise_conv_kernel_size (int): Kernel size of positionwise
                conv1d layer.
            macaron_style (bool): Whether to use macaron style for
                positionwise layer.
            selfattention_layer_type (str): Encoder attention layer type,
                the parameter has no effect now, it's just for configure
                compatibility.
            activation_type (str): Encoder activation function type.
            use_cnn_module (bool): Whether to use convolution module.
            cnn_module_kernel (int): Kernel size of convolution module.
            causal (bool): whether to use causal convolution or not.
            cnn_module_norm (str): cnn conv norm type, Optional['batch_norm','layer_norm']
        """
        assert check_argument_types()

        super().__init__(input_size, output_size, attention_heads, linear_units,
                         num_blocks, dropout_rate, positional_dropout_rate,
                         attention_dropout_rate, input_layer,
                         pos_enc_layer_type, normalize_before, concat_after,
                         static_chunk_size, use_dynamic_chunk, global_cmvn,
                         use_dynamic_left_chunk, max_len)
        activation = get_activation(activation_type)

        # self-attention module definition
        encoder_selfattn_layer = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args = (attention_heads, output_size,
                                       attention_dropout_rate)
        # feed-forward module definition
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (output_size, linear_units, dropout_rate,
                                   activation)
        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation,
                                  cnn_module_norm, causal)

        self.encoders = nn.LayerList([
            ConformerEncoderLayer(
                size=output_size,
                self_attn=encoder_selfattn_layer(*encoder_selfattn_layer_args),
                feed_forward=positionwise_layer(*positionwise_layer_args),
                feed_forward_macaron=positionwise_layer(
                    *positionwise_layer_args) if macaron_style else None,
                conv_module=convolution_layer(*convolution_layer_args)
                if use_cnn_module else None,
                dropout_rate=dropout_rate,
                normalize_before=normalize_before,
                concat_after=concat_after) for _ in range(num_blocks)
        ])


class SqueezeformerEncoder(nn.Layer):
    def __init__(self,
                 input_size: int,
                 encoder_dim: int=256,
                 output_size: int=256,
                 attention_heads: int=4,
                 num_blocks: int=12,
                 reduce_idx: Optional[Union[int, List[int]]]=5,
                 recover_idx: Optional[Union[int, List[int]]]=11,
                 feed_forward_expansion_factor: int=4,
                 dw_stride: bool=False,
                 input_dropout_rate: float=0.1,
                 pos_enc_layer_type: str="rel_pos",
                 time_reduction_layer_type: str="conv1d",
                 do_rel_shift: bool=True,
                 feed_forward_dropout_rate: float=0.1,
                 attention_dropout_rate: float=0.1,
                 cnn_module_kernel: int=31,
                 cnn_norm_type: str="layer_norm",
                 dropout: float=0.1,
                 causal: bool=False,
                 adaptive_scale: bool=True,
                 activation_type: str="swish",
                 init_weights: bool=True,
                 global_cmvn: paddle.nn.Layer=None,
                 normalize_before: bool=False,
                 use_dynamic_chunk: bool=False,
                 concat_after: bool=False,
                 static_chunk_size: int=0,
                 use_dynamic_left_chunk: bool=False):
        """Construct SqueezeformerEncoder

        Args:
            input_size to use_dynamic_chunk, see in Transformer BaseEncoder.
            encoder_dim (int): The hidden dimension of encoder layer.
            output_size (int): The output dimension of final projection layer.
            attention_heads (int): Num of attention head in attention module.
            num_blocks (int): Num of encoder layers.
            reduce_idx Optional[Union[int, List[int]]]:
                reduce layer index, from 40ms to 80ms per frame.
            recover_idx Optional[Union[int, List[int]]]:
                recover layer index, from 80ms to 40ms per frame.
            feed_forward_expansion_factor (int): Enlarge coefficient of FFN.
            dw_stride (bool): Whether do depthwise convolution
                              on subsampling module.
            input_dropout_rate (float): Dropout rate of input projection layer.
            pos_enc_layer_type (str): Self attention type.
            time_reduction_layer_type (str): Conv1d or Conv2d reduction layer.
            do_rel_shift (bool): Whether to do relative shift
                                 operation on rel-attention module.
            cnn_module_kernel (int): Kernel size of CNN module.
            activation_type (str): Encoder activation function type.
            cnn_module_kernel (int): Kernel size of convolution module.
            adaptive_scale (bool): Whether to use adaptive scale.
            init_weights (bool): Whether to initialize weights.
            causal (bool): whether to use causal convolution or not.
        """
        assert check_argument_types()
        super().__init__()
        self.global_cmvn = global_cmvn
        self.reduce_idx: Optional[Union[int, List[int]]] = [reduce_idx] \
            if type(reduce_idx) == int else reduce_idx
        self.recover_idx: Optional[Union[int, List[int]]] = [recover_idx] \
            if type(recover_idx) == int else recover_idx
        self.check_ascending_list()
        if reduce_idx is None:
            self.time_reduce = None
        else:
            if recover_idx is None:
                self.time_reduce = 'normal'  # no recovery at the end
            else:
                self.time_reduce = 'recover'  # recovery at the end
                assert len(self.reduce_idx) == len(self.recover_idx)
            self.reduce_stride = 2
        self._output_size = output_size
        self.normalize_before = normalize_before
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        activation = get_activation(activation_type)

        # self-attention module definition
        if pos_enc_layer_type != "rel_pos":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (attention_heads, output_size,
                                           attention_dropout_rate)
        else:
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (attention_heads, encoder_dim,
                                           attention_dropout_rate, do_rel_shift,
                                           adaptive_scale, init_weights)

        # feed-forward module definition
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            encoder_dim, encoder_dim * feed_forward_expansion_factor,
            feed_forward_dropout_rate, activation, adaptive_scale, init_weights)

        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (encoder_dim, cnn_module_kernel, activation,
                                  cnn_norm_type, causal, True, adaptive_scale,
                                  init_weights)

        self.embed = DepthwiseConv2DSubsampling4(
            1, encoder_dim,
            RelPositionalEncoding(encoder_dim, dropout_rate=0.1), dw_stride,
            input_size, input_dropout_rate, init_weights)

        self.preln = LayerNorm(encoder_dim)
        self.encoders = paddle.nn.LayerList([
            SqueezeformerEncoderLayer(
                encoder_dim,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                convolution_layer(*convolution_layer_args),
                positionwise_layer(*positionwise_layer_args), normalize_before,
                dropout, concat_after) for _ in range(num_blocks)
        ])
        if time_reduction_layer_type == 'conv1d':
            time_reduction_layer = TimeReductionLayer1D
            time_reduction_layer_args = {
                'channel': encoder_dim,
                'out_dim': encoder_dim,
            }
        elif time_reduction_layer_type == 'stream':
            time_reduction_layer = TimeReductionLayerStream
            time_reduction_layer_args = {
                'channel': encoder_dim,
                'out_dim': encoder_dim,
            }
        else:
            time_reduction_layer = TimeReductionLayer2D
            time_reduction_layer_args = {'encoder_dim': encoder_dim}

        self.time_reduction_layer = time_reduction_layer(
            **time_reduction_layer_args)
        self.time_recover_layer = Linear(encoder_dim, encoder_dim)
        self.final_proj = None
        if output_size != encoder_dim:
            self.final_proj = Linear(encoder_dim, output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(
            self,
            xs: paddle.Tensor,
            xs_lens: paddle.Tensor,
            decoding_chunk_size: int=0,
            num_decoding_left_chunks: int=-1,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Embed positions in tensor.
        Args:
            xs: padded input tensor (B, L, D)
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
            encoder output tensor, lens and mask
        """
        masks = make_non_pad_mask(xs_lens).unsqueeze(1)  # (B, 1, L)

        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks
        chunk_masks = add_optional_chunk_mask(
            xs, masks, self.use_dynamic_chunk, self.use_dynamic_left_chunk,
            decoding_chunk_size, self.static_chunk_size,
            num_decoding_left_chunks)
        xs_lens = chunk_masks.squeeze(1).sum(1)
        xs = self.preln(xs)
        recover_activations: \
            List[Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]] = []
        index = 0
        for i, layer in enumerate(self.encoders):
            if self.reduce_idx is not None:
                if self.time_reduce is not None and i in self.reduce_idx:
                    recover_activations.append(
                        (xs, chunk_masks, pos_emb, mask_pad))
                    xs, xs_lens, chunk_masks, mask_pad = self.time_reduction_layer(
                        xs, xs_lens, chunk_masks, mask_pad)
                    pos_emb = pos_emb[:, ::2, :]
                    index += 1

            if self.recover_idx is not None:
                if self.time_reduce == 'recover' and i in self.recover_idx:
                    index -= 1
                    recover_tensor, recover_chunk_masks, recover_pos_emb, recover_mask_pad = recover_activations[
                        index]
                    # recover output length for ctc decode
                    xs = paddle.repeat_interleave(xs, repeats=2, axis=1)
                    xs = self.time_recover_layer(xs)
                    recoverd_t = recover_tensor.shape[1]
                    xs = recover_tensor + xs[:, :recoverd_t, :]
                    chunk_masks = recover_chunk_masks
                    pos_emb = recover_pos_emb
                    mask_pad = recover_mask_pad

            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)

        if self.final_proj is not None:
            xs = self.final_proj(xs)
        return xs, masks

    def check_ascending_list(self):
        if self.reduce_idx is not None:
            assert self.reduce_idx == sorted(self.reduce_idx), \
                "reduce_idx should be int or ascending list"
        if self.recover_idx is not None:
            assert self.recover_idx == sorted(self.recover_idx), \
                "recover_idx should be int or ascending list"

    def calculate_downsampling_factor(self, i: int) -> int:
        if self.reduce_idx is None:
            return 1
        else:
            reduce_exp, recover_exp = 0, 0
            for exp, rd_idx in enumerate(self.reduce_idx):
                if i >= rd_idx:
                    reduce_exp = exp + 1
            if self.recover_idx is not None:
                for exp, rc_idx in enumerate(self.recover_idx):
                    if i >= rc_idx:
                        recover_exp = exp + 1
            return int(2**(reduce_exp - recover_exp))

    def forward_chunk(
            self,
            xs: paddle.Tensor,
            offset: int,
            required_cache_size: int,
            att_cache: paddle.Tensor=paddle.zeros([0, 0, 0, 0]),
            cnn_cache: paddle.Tensor=paddle.zeros([0, 0, 0, 0]),
            att_mask: paddle.Tensor=paddle.ones([0, 0, 0], dtype=paddle.bool),
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """ Forward just one chunk

        Args:
            xs (paddle.Tensor): chunk input, with shape (b=1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            att_cache (paddle.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (paddle.Tensor): cache tensor for cnn_module in conformer,
                (elayers, b=1, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`

        Returns:
            paddle.Tensor: output of current input xs,
                with shape (b=1, chunk_size, hidden-dim).
            paddle.Tensor: new attention cache required for next chunk, with
                dynamic shape (elayers, head, ?, d_k * 2)
                depending on required_cache_size.
            paddle.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.
        """
        assert xs.shape[0] == 1  # batch size must be one

        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)

        # tmp_masks is just for interface compatibility, [B=1, C=1, T]
        tmp_masks = paddle.ones([1, 1, xs.shape[1]], dtype=paddle.bool)
        # before embed, xs=(B, T, D1), pos_emb=(B=1, T, D)
        xs, pos_emb, _ = self.embed(xs, tmp_masks, offset=offset)

        # NOTE(xcsong): After  embed, shape(xs) is (b=1, chunk_size, hidden-dim)
        elayers, cache_t1 = att_cache.shape[0], att_cache.shape[2]
        chunk_size = xs.shape[1]
        attention_key_size = cache_t1 + chunk_size
        pos_emb = self.embed.position_encoding(
            offset=offset - cache_t1, size=attention_key_size)
        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = attention_key_size
        else:
            next_cache_start = max(attention_key_size - required_cache_size, 0)

        r_att_cache = []
        r_cnn_cache = []

        mask_pad = paddle.ones([1, xs.shape[1]], dtype=paddle.bool)
        mask_pad = mask_pad.unsqueeze(1)
        max_att_len: int = 0
        recover_activations: \
            List[Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]] = []
        index = 0
        xs_lens = paddle.to_tensor([xs.shape[1]], dtype=paddle.int32)
        xs = self.preln(xs)
        for i, layer in enumerate(self.encoders):
            # NOTE(xcsong): Before layer.forward
            #   shape(att_cache[i:i + 1]) is (1, head, cache_t1, d_k * 2),
            #   shape(cnn_cache[i])       is (b=1, hidden-dim, cache_t2)
            if self.reduce_idx is not None:
                if self.time_reduce is not None and i in self.reduce_idx:
                    recover_activations.append(
                        (xs, att_mask, pos_emb, mask_pad))
                    xs, xs_lens, att_mask, mask_pad = self.time_reduction_layer(
                        xs, xs_lens, att_mask, mask_pad)
                    pos_emb = pos_emb[:, ::2, :]
                    index += 1

            if self.recover_idx is not None:
                if self.time_reduce == 'recover' and i in self.recover_idx:
                    index -= 1
                    recover_tensor, recover_att_mask, recover_pos_emb, recover_mask_pad = recover_activations[
                        index]
                    # recover output length for ctc decode
                    xs = paddle.repeat_interleave(xs, repeats=2, axis=1)
                    xs = self.time_recover_layer(xs)
                    recoverd_t = recover_tensor.shape[1]
                    xs = recover_tensor + xs[:, :recoverd_t, :]
                    att_mask = recover_att_mask
                    pos_emb = recover_pos_emb
                    mask_pad = recover_mask_pad

            factor = self.calculate_downsampling_factor(i)
            att_cache1 = att_cache[
                i:i + 1][:, :, ::factor, :][:, :, :pos_emb.shape[1] - xs.shape[
                    1], :]
            cnn_cache1 = cnn_cache[i] if cnn_cache.shape[0] > 0 else cnn_cache
            xs, _, new_att_cache, new_cnn_cache = layer(
                xs,
                att_mask,
                pos_emb,
                att_cache=att_cache1,
                cnn_cache=cnn_cache1)
            # NOTE(xcsong): After layer.forward
            #   shape(new_att_cache) is (1, head, attention_key_size, d_k * 2),
            #   shape(new_cnn_cache) is (b=1, hidden-dim, cache_t2)
            cached_att = new_att_cache[:, :, next_cache_start // factor:, :]
            cached_cnn = new_cnn_cache.unsqueeze(0)
            cached_att = cached_att.repeat_interleave(repeats=factor, axis=2)
            if i == 0:
                # record length for the first block as max length
                max_att_len = cached_att.shape[2]
            r_att_cache.append(cached_att[:, :, :max_att_len, :])
            r_cnn_cache.append(cached_cnn)
        # NOTE(xcsong): shape(r_att_cache) is (elayers, head, ?, d_k * 2),
        #   ? may be larger than cache_t1, it depends on required_cache_size
        r_att_cache = paddle.concat(r_att_cache, axis=0)
        # NOTE(xcsong): shape(r_cnn_cache) is (e, b=1, hidden-dim, cache_t2)
        r_cnn_cache = paddle.concat(r_cnn_cache, axis=0)

        if self.final_proj is not None:
            xs = self.final_proj(xs)
        return xs, r_att_cache, r_cnn_cache
