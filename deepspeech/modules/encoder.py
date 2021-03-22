# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""Encoder definition."""
import logging
from typing import Tuple, List, Optional
from typeguard import check_argument_types

import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn import initializer as I

from deepspeech.modules.attention import MultiHeadedAttention
from deepspeech.modules.attention import RelPositionMultiHeadedAttention
from deepspeech.modules.convolution import ConvolutionModule
from deepspeech.modules.embedding import PositionalEncoding
from deepspeech.modules.embedding import RelPositionalEncoding
from deepspeech.modules.encoder_layer import TransformerEncoderLayer
from deepspeech.modules.encoder_layer import ConformerEncoderLayer
from deepspeech.modules.positionwise_feed_forward import PositionwiseFeedForward
from deepspeech.modules.subsampling import Conv2dSubsampling4
from deepspeech.modules.subsampling import Conv2dSubsampling6
from deepspeech.modules.subsampling import Conv2dSubsampling8
from deepspeech.modules.subsampling import LinearNoSubsampling
from deepspeech.modules.mask import make_pad_mask
from deepspeech.modules.mask import add_optional_chunk_mask
from deepspeech.modules.activation import get_activation

logger = logging.getLogger(__name__)

__all__ = ["BaseEncoder", 'TransformerEncoder', "ConformerEncoder"]


class BaseEncoder(nn.Layer):
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
            global_cmvn: torch.nn.Module=None,
            use_dynamic_left_chunk: bool=False, ):
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
                opitonal [abs_pos, scaled_abs_pos, rel_pos]
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
            global_cmvn (Optional[torch.nn.Module]): Optional GlobalCMVN module
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
            input_size,
            output_size,
            dropout_rate,
            pos_enc_class(output_size, positional_dropout_rate), )

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-12)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk

    def output_size(self) -> int:
        return self._output_size

    def forward(
            self,
            xs: torch.Tensor,
            xs_lens: torch.Tensor,
            decoding_chunk_size: int=0,
            num_decoding_left_chunks: int=-1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        masks = ~make_pad_mask(xs_lens).unsqueeze(1)  # (B, 1, L)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = ~masks
        chunk_masks = add_optional_chunk_mask(
            xs, masks, self.use_dynamic_chunk, self.use_dynamic_left_chunk,
            decoding_chunk_size, self.static_chunk_size,
            num_decoding_left_chunks)
        for layer in self.encoders:
            xs, chunk_masks, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    def forward_chunk(
            self,
            xs: torch.Tensor,
            offset: int,
            required_cache_size: int,
            subsampling_cache: Optional[torch.Tensor]=None,
            elayers_output_cache: Optional[List[torch.Tensor]]=None,
            conformer_cnn_cache: Optional[List[torch.Tensor]]=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[
            torch.Tensor]]:
        """ Forward just one chunk
        Args:
            xs (torch.Tensor): chunk input
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            subsampling_cache (Optional[torch.Tensor]): subsampling cache
            elayers_output_cache (Optional[List[torch.Tensor]]):
                transformer/conformer encoder layers output cache
            conformer_cnn_cache (Optional[List[torch.Tensor]]): conformer
                cnn cache
        Returns:
            torch.Tensor: output of current input xs
            torch.Tensor: subsampling cache required for next chunk computation
            List[torch.Tensor]: encoder layers output cache required for next
                chunk computation
            List[torch.Tensor]: conformer cnn cache
        """
        assert xs.size(0) == 1
        # tmp_masks is just for interface compatibility
        tmp_masks = torch.ones(
            1, xs.size(1), device=xs.device, dtype=torch.bool)
        tmp_masks = tmp_masks.unsqueeze(1)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, _ = self.embed(xs, tmp_masks, offset)
        if subsampling_cache is not None:
            cache_size = subsampling_cache.size(1)
            xs = torch.cat((subsampling_cache, xs), dim=1)
        else:
            cache_size = 0
        pos_emb = self.embed.position_encoding(offset - cache_size, xs.size(1))
        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = xs.size(1)
        else:
            next_cache_start = xs.size(1) - required_cache_size
        r_subsampling_cache = xs[:, next_cache_start:, :]
        # Real mask for transformer/conformer layers
        masks = torch.ones(1, xs.size(1), device=xs.device, dtype=torch.bool)
        masks = masks.unsqueeze(1)
        r_elayers_output_cache = []
        r_conformer_cnn_cache = []
        for i, layer in enumerate(self.encoders):
            if elayers_output_cache is None:
                attn_cache = None
            else:
                attn_cache = elayers_output_cache[i]
            if conformer_cnn_cache is None:
                cnn_cache = None
            else:
                cnn_cache = conformer_cnn_cache[i]
            xs, _, new_cnn_cache = layer(
                xs,
                masks,
                pos_emb,
                output_cache=attn_cache,
                cnn_cache=cnn_cache)
            r_elayers_output_cache.append(xs[:, next_cache_start:, :])
            r_conformer_cnn_cache.append(new_cnn_cache)
        if self.normalize_before:
            xs = self.after_norm(xs)

        return (xs[:, cache_size:, :], r_subsampling_cache,
                r_elayers_output_cache, r_conformer_cnn_cache)

    def forward_chunk_by_chunk(
            self,
            xs: torch.Tensor,
            decoding_chunk_size: int,
            num_decoding_left_chunks: int=-1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            xs (torch.Tensor): (1, max_len, dim)
            chunk_size (int): decoding chunk size
        """
        assert decoding_chunk_size > 0
        # The model is trained by static or dynamic chunk
        assert self.static_chunk_size > 0 or self.use_dynamic_chunk
        subsampling = self.embed.subsampling_rate
        context = self.embed.right_context + 1  # Add current frame
        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        num_frames = xs.size(1)
        subsampling_cache: Optional[torch.Tensor] = None
        elayers_output_cache: Optional[List[torch.Tensor]] = None
        conformer_cnn_cache: Optional[List[torch.Tensor]] = None
        outputs = []
        offset = 0
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks

        # Feed forward overlap input step by step
        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + decoding_window, num_frames)
            chunk_xs = xs[:, cur:end, :]
            (y, subsampling_cache, elayers_output_cache,
             conformer_cnn_cache) = self.forward_chunk(
                 chunk_xs, offset, required_cache_size, subsampling_cache,
                 elayers_output_cache, conformer_cnn_cache)
            outputs.append(y)
            offset += y.size(1)
        ys = torch.cat(outputs, 1)
        masks = torch.ones(1, ys.size(1), device=ys.device, dtype=torch.bool)
        masks = masks.unsqueeze(1)
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
            global_cmvn: torch.nn.Module=None,
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
        self.encoders = torch.nn.ModuleList([
            TransformerEncoderLayer(
                output_size,
                MultiHeadedAttention(attention_heads, output_size,
                                     attention_dropout_rate),
                PositionwiseFeedForward(output_size, linear_units,
                                        dropout_rate), dropout_rate,
                normalize_before, concat_after) for _ in range(num_blocks)
        ])


class ConformerEncoder(BaseEncoder):
    """Conformer encoder module."""

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
            pos_enc_layer_type: str="rel_pos",
            normalize_before: bool=True,
            concat_after: bool=False,
            static_chunk_size: int=0,
            use_dynamic_chunk: bool=False,
            global_cmvn: torch.nn.Module=None,
            use_dynamic_left_chunk: bool=False,
            positionwise_conv_kernel_size: int=1,
            macaron_style: bool=True,
            selfattention_layer_type: str="rel_selfattn",
            activation_type: str="swish",
            use_cnn_module: bool=True,
            cnn_module_kernel: int=15,
            causal: bool=False,
            cnn_module_norm: str="batch_norm", ):
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
        """
        assert check_argument_types()
        super().__init__(input_size, output_size, attention_heads, linear_units,
                         num_blocks, dropout_rate, positional_dropout_rate,
                         attention_dropout_rate, input_layer,
                         pos_enc_layer_type, normalize_before, concat_after,
                         static_chunk_size, use_dynamic_chunk, global_cmvn,
                         use_dynamic_left_chunk)
        activation = get_activation(activation_type)

        # self-attention module definition
        encoder_selfattn_layer = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args = (attention_heads, output_size,
                                       attention_dropout_rate, )
        # feed-forward module definition
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (output_size, linear_units, dropout_rate,
                                   activation, )
        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation,
                                  cnn_module_norm, causal)

        self.encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args)
                if macaron_style else None,
                convolution_layer(*convolution_layer_args)
                if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after, ) for _ in range(num_blocks)
        ])
