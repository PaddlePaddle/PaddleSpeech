# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from typing import Dict
from typing import List
from typing import Optional

import paddle
from paddle import nn

from paddlespeech.t2s.modules.activation import get_activation
from paddlespeech.t2s.modules.conformer.convolution import ConvolutionModule
from paddlespeech.t2s.modules.conformer.encoder_layer import EncoderLayer
from paddlespeech.t2s.modules.layer_norm import LayerNorm
from paddlespeech.t2s.modules.masked_fill import masked_fill
from paddlespeech.t2s.modules.nets_utils import initialize
from paddlespeech.t2s.modules.tacotron2.decoder import Postnet
from paddlespeech.t2s.modules.transformer.attention import LegacyRelPositionMultiHeadedAttention
from paddlespeech.t2s.modules.transformer.attention import MultiHeadedAttention
from paddlespeech.t2s.modules.transformer.attention import RelPositionMultiHeadedAttention
from paddlespeech.t2s.modules.transformer.embedding import LegacyRelPositionalEncoding
from paddlespeech.t2s.modules.transformer.embedding import PositionalEncoding
from paddlespeech.t2s.modules.transformer.embedding import RelPositionalEncoding
from paddlespeech.t2s.modules.transformer.embedding import ScaledPositionalEncoding
from paddlespeech.t2s.modules.transformer.multi_layer_conv import Conv1dLinear
from paddlespeech.t2s.modules.transformer.multi_layer_conv import MultiLayeredConv1d
from paddlespeech.t2s.modules.transformer.positionwise_feed_forward import PositionwiseFeedForward
from paddlespeech.t2s.modules.transformer.repeat import repeat
from paddlespeech.t2s.modules.transformer.subsampling import Conv2dSubsampling


# MLM -> Mask Language Model
class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._sub_layers.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class MaskInputLayer(nn.Layer):
    def __init__(self, out_features: int) -> None:
        super().__init__()
        self.mask_feature = paddle.create_parameter(
            shape=(1, 1, out_features),
            dtype=paddle.float32,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.normal(shape=(1, 1, out_features))))

    def forward(self, input: paddle.Tensor,
                masked_pos: paddle.Tensor=None) -> paddle.Tensor:
        masked_pos = paddle.expand_as(paddle.unsqueeze(masked_pos, -1), input)
        masked_input = masked_fill(input, masked_pos, 0) + masked_fill(
            paddle.expand_as(self.mask_feature, input), ~masked_pos, 0)
        return masked_input


class MLMEncoder(nn.Layer):
    """Conformer encoder module.

    Args:
        idim (int): 
            Input dimension.
        attention_dim (int): 
            Dimension of attention.
        attention_heads (int): 
            The number of heads of multi head attention.
        linear_units (int): 
            The number of units of position-wise feed forward.
        num_blocks (int): 
            The number of decoder blocks.
        dropout_rate (float): 
            Dropout rate.
        positional_dropout_rate (float): 
            Dropout rate after adding positional encoding.
        attention_dropout_rate (float): 
            Dropout rate in attention.
        input_layer (Union[str, paddle.nn.Layer]): 
            Input layer type.
        normalize_before (bool): 
            Whether to use layer_norm before the first block.
        concat_after (bool): 
            Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): 
            "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): 
            Kernel size of positionwise conv1d layer.
        macaron_style (bool): 
            Whether to use macaron style for positionwise layer.
        pos_enc_layer_type (str): 
            Encoder positional encoding layer type.
        selfattention_layer_type (str): 
            Encoder attention layer type.
        activation_type (str): 
            Encoder activation function type.
        use_cnn_module (bool): 
            Whether to use convolution module.
        zero_triu (bool): 
            Whether to zero the upper triangular part of attention matrix.
        cnn_module_kernel (int): 
            Kernerl size of convolution module.
        padding_idx (int): 
            Padding idx for input_layer=embed.
        stochastic_depth_rate (float): 
            Maximum probability to skip the encoder layer.

    """

    def __init__(self,
                 idim: int,
                 vocab_size: int=0,
                 pre_speech_layer: int=0,
                 attention_dim: int=256,
                 attention_heads: int=4,
                 linear_units: int=2048,
                 num_blocks: int=6,
                 dropout_rate: float=0.1,
                 positional_dropout_rate: float=0.1,
                 attention_dropout_rate: float=0.0,
                 input_layer: str="conv2d",
                 normalize_before: bool=True,
                 concat_after: bool=False,
                 positionwise_layer_type: str="linear",
                 positionwise_conv_kernel_size: int=1,
                 macaron_style: bool=False,
                 pos_enc_layer_type: str="abs_pos",
                 pos_enc_class=None,
                 selfattention_layer_type: str="selfattn",
                 activation_type: str="swish",
                 use_cnn_module: bool=False,
                 zero_triu: bool=False,
                 cnn_module_kernel: int=31,
                 padding_idx: int=-1,
                 stochastic_depth_rate: float=0.0,
                 text_masking: bool=False):
        """Construct an Encoder object."""
        super().__init__()
        self._output_size = attention_dim
        self.text_masking = text_masking
        if self.text_masking:
            self.text_masking_layer = MaskInputLayer(attention_dim)
        activation = get_activation(activation_type)
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            pos_enc_class = LegacyRelPositionalEncoding
            assert selfattention_layer_type == "legacy_rel_selfattn"
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        self.conv_subsampling_factor = 1
        if input_layer == "linear":
            self.embed = nn.Sequential(
                nn.Linear(idim, attention_dim),
                nn.LayerNorm(attention_dim),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate), )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                idim,
                attention_dim,
                dropout_rate,
                pos_enc_class(attention_dim, positional_dropout_rate), )
            self.conv_subsampling_factor = 4
        elif input_layer == "embed":
            self.embed = nn.Sequential(
                nn.Embedding(idim, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate), )
        elif input_layer == "mlm":
            self.segment_emb = None
            self.speech_embed = mySequential(
                MaskInputLayer(idim),
                nn.Linear(idim, attention_dim),
                nn.LayerNorm(attention_dim),
                nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate))
            self.text_embed = nn.Sequential(
                nn.Embedding(
                    vocab_size, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate), )
        elif input_layer == "sega_mlm":
            self.segment_emb = nn.Embedding(
                500, attention_dim, padding_idx=padding_idx)
            self.speech_embed = mySequential(
                MaskInputLayer(idim),
                nn.Linear(idim, attention_dim),
                nn.LayerNorm(attention_dim),
                nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate))
            self.text_embed = nn.Sequential(
                nn.Embedding(
                    vocab_size, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate), )
        elif isinstance(input_layer, nn.Layer):
            self.embed = nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate), )
        elif input_layer is None:
            self.embed = nn.Sequential(
                pos_enc_class(attention_dim, positional_dropout_rate))
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before

        # self-attention module definition
        if selfattention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (attention_heads, attention_dim,
                                           attention_dropout_rate, )
        elif selfattention_layer_type == "legacy_rel_selfattn":
            assert pos_enc_layer_type == "legacy_rel_pos"
            encoder_selfattn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (attention_heads, attention_dim,
                                           attention_dropout_rate, )
        elif selfattention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (attention_heads, attention_dim,
                                           attention_dropout_rate, zero_triu, )
        else:
            raise ValueError("unknown encoder_attn_layer: " +
                             selfattention_layer_type)

        # feed-forward module definition
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units,
                                       dropout_rate, activation, )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (attention_dim, linear_units,
                                       positionwise_conv_kernel_size,
                                       dropout_rate, )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (attention_dim, linear_units,
                                       positionwise_conv_kernel_size,
                                       dropout_rate, )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel, activation)

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                attention_dim,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                stochastic_depth_rate * float(1 + lnum) / num_blocks, ), )
        self.pre_speech_layer = pre_speech_layer
        self.pre_speech_encoders = repeat(
            self.pre_speech_layer,
            lambda lnum: EncoderLayer(
                attention_dim,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                stochastic_depth_rate * float(1 + lnum) / self.pre_speech_layer, ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def forward(self,
                speech: paddle.Tensor,
                text: paddle.Tensor,
                masked_pos: paddle.Tensor,
                speech_mask: paddle.Tensor=None,
                text_mask: paddle.Tensor=None,
                speech_seg_pos: paddle.Tensor=None,
                text_seg_pos: paddle.Tensor=None):
        """Encode input sequence.

        """
        if masked_pos is not None:
            speech = self.speech_embed(speech, masked_pos)
        else:
            speech = self.speech_embed(speech)
        if text is not None:
            text = self.text_embed(text)
        if speech_seg_pos is not None and text_seg_pos is not None and self.segment_emb:
            speech_seg_emb = self.segment_emb(speech_seg_pos)
            text_seg_emb = self.segment_emb(text_seg_pos)
            text = (text[0] + text_seg_emb, text[1])
            speech = (speech[0] + speech_seg_emb, speech[1])
        if self.pre_speech_encoders:
            speech, _ = self.pre_speech_encoders(speech, speech_mask)

        if text is not None:
            xs = paddle.concat([speech[0], text[0]], axis=1)
            xs_pos_emb = paddle.concat([speech[1], text[1]], axis=1)
            masks = paddle.concat([speech_mask, text_mask], axis=-1)
        else:
            xs = speech[0]
            xs_pos_emb = speech[1]
            masks = speech_mask

        xs, masks = self.encoders((xs, xs_pos_emb), masks)

        if isinstance(xs, tuple):
            xs = xs[0]
        if self.normalize_before:
            xs = self.after_norm(xs)

        return xs, masks


class MLMDecoder(MLMEncoder):
    def forward(self, xs: paddle.Tensor, masks: paddle.Tensor):
        """Encode input sequence.

        Args:
            xs (paddle.Tensor): 
                Input tensor (#batch, time, idim).
            masks (paddle.Tensor): 
                Mask tensor (#batch, time).

        Returns:
            paddle.Tensor: 
                Output tensor (#batch, time, attention_dim).
            paddle.Tensor: 
                Mask tensor (#batch, time).

        """
        xs = self.embed(xs)
        xs, masks = self.encoders(xs, masks)

        if isinstance(xs, tuple):
            xs = xs[0]
        if self.normalize_before:
            xs = self.after_norm(xs)

        return xs, masks


# encoder and decoder is nn.Layer, not str
class MLM(nn.Layer):
    def __init__(self,
                 odim: int,
                 encoder: nn.Layer,
                 decoder: Optional[nn.Layer],
                 postnet_layers: int=0,
                 postnet_chans: int=0,
                 postnet_filts: int=0,
                 text_masking: bool=False):

        super().__init__()
        self.odim = odim
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = encoder.text_embed[0]._num_embeddings

        if self.decoder is None or not (hasattr(self.decoder,
                                                'output_layer') and
                                        self.decoder.output_layer is not None):
            self.sfc = nn.Linear(self.encoder._output_size, odim)
        else:
            self.sfc = None
        if text_masking:
            self.text_sfc = nn.Linear(
                self.encoder.text_embed[0]._embedding_dim,
                self.vocab_size,
                weight_attr=self.encoder.text_embed[0]._weight_attr)
        else:
            self.text_sfc = None

        self.postnet = (None if postnet_layers == 0 else Postnet(
            idim=self.encoder._output_size,
            odim=odim,
            n_layers=postnet_layers,
            n_chans=postnet_chans,
            n_filts=postnet_filts,
            use_batch_norm=True,
            dropout_rate=0.5, ))

    def inference(
            self,
            speech: paddle.Tensor,
            text: paddle.Tensor,
            masked_pos: paddle.Tensor,
            speech_mask: paddle.Tensor,
            text_mask: paddle.Tensor,
            speech_seg_pos: paddle.Tensor,
            text_seg_pos: paddle.Tensor,
            span_bdy: List[int],
            use_teacher_forcing: bool=True, ) -> List[paddle.Tensor]:
        '''
        Args:
            speech (paddle.Tensor): 
                input speech (1, Tmax, D).
            text (paddle.Tensor): 
                input text (1, Tmax2).
            masked_pos (paddle.Tensor): 
                masked position of input speech (1, Tmax)
            speech_mask (paddle.Tensor): 
                mask of speech (1, 1, Tmax).
            text_mask (paddle.Tensor): 
                mask of text (1, 1, Tmax2).
            speech_seg_pos (paddle.Tensor): 
                n-th phone of each mel, 0<=n<=Tmax2 (1, Tmax).
            text_seg_pos (paddle.Tensor): 
                n-th phone of each phone, 0<=n<=Tmax2 (1, Tmax2).
            span_bdy (List[int]): 
                masked mel boundary of input speech (2,)
            use_teacher_forcing (bool): 
                whether to use teacher forcing
        Returns:
            List[Tensor]:
                eg: [Tensor(shape=[1, 181, 80]), Tensor(shape=[80, 80]), Tensor(shape=[1, 67, 80])]
        '''

        z_cache = None
        if use_teacher_forcing:
            before_outs, zs, *_ = self.forward(
                speech=speech,
                text=text,
                masked_pos=masked_pos,
                speech_mask=speech_mask,
                text_mask=text_mask,
                speech_seg_pos=speech_seg_pos,
                text_seg_pos=text_seg_pos)
            if zs is None:
                zs = before_outs

            speech = speech.squeeze(0)
            outs = [speech[:span_bdy[0]]]
            outs += [zs[0][span_bdy[0]:span_bdy[1]]]
            outs += [speech[span_bdy[1]:]]
            return outs
        return None


class MLMEncAsDecoder(MLM):
    def forward(self,
                speech: paddle.Tensor,
                text: paddle.Tensor,
                masked_pos: paddle.Tensor,
                speech_mask: paddle.Tensor,
                text_mask: paddle.Tensor,
                speech_seg_pos: paddle.Tensor,
                text_seg_pos: paddle.Tensor):
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out, h_masks = self.encoder(
            speech=speech,
            text=text,
            masked_pos=masked_pos,
            speech_mask=speech_mask,
            text_mask=text_mask,
            speech_seg_pos=speech_seg_pos,
            text_seg_pos=text_seg_pos)
        if self.decoder is not None:
            zs, _ = self.decoder(encoder_out, h_masks)
        else:
            zs = encoder_out
        speech_hidden_states = zs[:, :paddle.shape(speech)[1:2], :]
        if self.sfc is not None:
            before_outs = paddle.reshape(
                self.sfc(speech_hidden_states),
                (paddle.shape(speech_hidden_states)[0:1], -1, self.odim))
        else:
            before_outs = speech_hidden_states
        if self.postnet is not None:
            after_outs = before_outs + paddle.transpose(
                self.postnet(paddle.transpose(before_outs, [0, 2, 1])),
                [0, 2, 1])
        else:
            after_outs = None
        return before_outs, after_outs, None


class MLMDualMaksing(MLM):
    def forward(self,
                speech: paddle.Tensor,
                text: paddle.Tensor,
                masked_pos: paddle.Tensor,
                speech_mask: paddle.Tensor,
                text_mask: paddle.Tensor,
                speech_seg_pos: paddle.Tensor,
                text_seg_pos: paddle.Tensor):
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out, h_masks = self.encoder(
            speech=speech,
            text=text,
            masked_pos=masked_pos,
            speech_mask=speech_mask,
            text_mask=text_mask,
            speech_seg_pos=speech_seg_pos,
            text_seg_pos=text_seg_pos)
        if self.decoder is not None:
            zs, _ = self.decoder(encoder_out, h_masks)
        else:
            zs = encoder_out
        speech_hidden_states = zs[:, :paddle.shape(speech)[1:2], :]
        if self.text_sfc:
            text_hiddent_states = zs[:, paddle.shape(speech)[1:2]:, :]
            text_outs = paddle.reshape(
                self.text_sfc(text_hiddent_states),
                (paddle.shape(text_hiddent_states)[0:1], -1, self.vocab_size))
        if self.sfc is not None:
            before_outs = paddle.reshape(
                self.sfc(speech_hidden_states),
                (paddle.shape(speech_hidden_states)[0:1], -1, self.odim))
        else:
            before_outs = speech_hidden_states
        if self.postnet is not None:
            after_outs = before_outs + paddle.transpose(
                self.postnet(paddle.transpose(before_outs, [0, 2, 1])),
                [0, 2, 1])
        else:
            after_outs = None
        return before_outs, after_outs, text_outs


class ErnieSAT(nn.Layer):
    def __init__(
            self,
            # network structure related
            idim: int,
            odim: int,
            postnet_layers: int=5,
            postnet_filts: int=5,
            postnet_chans: int=256,
            use_scaled_pos_enc: bool=False,
            encoder_type: str='conformer',
            decoder_type: str='conformer',
            enc_input_layer: str='sega_mlm',
            enc_pre_speech_layer: int=0,
            enc_cnn_module_kernel: int=7,
            enc_attention_dim: int=384,
            enc_attention_heads: int=2,
            enc_linear_units: int=1536,
            enc_num_blocks: int=4,
            enc_dropout_rate: float=0.2,
            enc_positional_dropout_rate: float=0.2,
            enc_attention_dropout_rate: float=0.2,
            enc_normalize_before: bool=True,
            enc_macaron_style: bool=True,
            enc_use_cnn_module: bool=True,
            enc_selfattention_layer_type: str='legacy_rel_selfattn',
            enc_activation_type: str='swish',
            enc_pos_enc_layer_type: str='legacy_rel_pos',
            enc_positionwise_layer_type: str='conv1d',
            enc_positionwise_conv_kernel_size: int=3,
            text_masking: bool=False,
            dec_cnn_module_kernel: int=31,
            dec_attention_dim: int=384,
            dec_attention_heads: int=2,
            dec_linear_units: int=1536,
            dec_num_blocks: int=4,
            dec_dropout_rate: float=0.2,
            dec_positional_dropout_rate: float=0.2,
            dec_attention_dropout_rate: float=0.2,
            dec_macaron_style: bool=True,
            dec_use_cnn_module: bool=True,
            dec_selfattention_layer_type: str='legacy_rel_selfattn',
            dec_activation_type: str='swish',
            dec_pos_enc_layer_type: str='legacy_rel_pos',
            dec_positionwise_layer_type: str='conv1d',
            dec_positionwise_conv_kernel_size: int=3,
            init_type: str="xavier_uniform", ):
        super().__init__()
        # store hyperparameters
        self.odim = odim

        self.use_scaled_pos_enc = use_scaled_pos_enc

        # initialize parameters
        initialize(self, init_type)

        # Encoder
        if encoder_type == "conformer":
            encoder = MLMEncoder(
                idim=odim,
                vocab_size=idim,
                pre_speech_layer=enc_pre_speech_layer,
                attention_dim=enc_attention_dim,
                attention_heads=enc_attention_heads,
                linear_units=enc_linear_units,
                num_blocks=enc_num_blocks,
                dropout_rate=enc_dropout_rate,
                positional_dropout_rate=enc_positional_dropout_rate,
                attention_dropout_rate=enc_attention_dropout_rate,
                input_layer=enc_input_layer,
                normalize_before=enc_normalize_before,
                positionwise_layer_type=enc_positionwise_layer_type,
                positionwise_conv_kernel_size=enc_positionwise_conv_kernel_size,
                macaron_style=enc_macaron_style,
                pos_enc_layer_type=enc_pos_enc_layer_type,
                selfattention_layer_type=enc_selfattention_layer_type,
                activation_type=enc_activation_type,
                use_cnn_module=enc_use_cnn_module,
                cnn_module_kernel=enc_cnn_module_kernel,
                text_masking=text_masking)
        else:
            raise ValueError(f"{encoder_type} is not supported.")

        # Decoder
        if decoder_type != 'no_decoder':
            decoder = MLMDecoder(
                idim=0,
                input_layer=None,
                cnn_module_kernel=dec_cnn_module_kernel,
                attention_dim=dec_attention_dim,
                attention_heads=dec_attention_heads,
                linear_units=dec_linear_units,
                num_blocks=dec_num_blocks,
                dropout_rate=dec_dropout_rate,
                positional_dropout_rate=dec_positional_dropout_rate,
                macaron_style=dec_macaron_style,
                use_cnn_module=dec_use_cnn_module,
                selfattention_layer_type=dec_selfattention_layer_type,
                activation_type=dec_activation_type,
                pos_enc_layer_type=dec_pos_enc_layer_type,
                positionwise_layer_type=dec_positionwise_layer_type,
                positionwise_conv_kernel_size=dec_positionwise_conv_kernel_size)

        else:
            decoder = None

        model_class = MLMDualMaksing if text_masking else MLMEncAsDecoder

        self.model = model_class(
            odim=odim,
            encoder=encoder,
            decoder=decoder,
            postnet_layers=postnet_layers,
            postnet_filts=postnet_filts,
            postnet_chans=postnet_chans,
            text_masking=text_masking)

        nn.initializer.set_global_initializer(None)

    def forward(self,
                speech: paddle.Tensor,
                text: paddle.Tensor,
                masked_pos: paddle.Tensor,
                speech_mask: paddle.Tensor,
                text_mask: paddle.Tensor,
                speech_seg_pos: paddle.Tensor,
                text_seg_pos: paddle.Tensor):
        return self.model(
            speech=speech,
            text=text,
            masked_pos=masked_pos,
            speech_mask=speech_mask,
            text_mask=text_mask,
            speech_seg_pos=speech_seg_pos,
            text_seg_pos=text_seg_pos)

    def inference(
            self,
            speech: paddle.Tensor,
            text: paddle.Tensor,
            masked_pos: paddle.Tensor,
            speech_mask: paddle.Tensor,
            text_mask: paddle.Tensor,
            speech_seg_pos: paddle.Tensor,
            text_seg_pos: paddle.Tensor,
            span_bdy: List[int],
            use_teacher_forcing: bool=True, ) -> Dict[str, paddle.Tensor]:
        return self.model.inference(
            speech=speech,
            text=text,
            masked_pos=masked_pos,
            speech_mask=speech_mask,
            text_mask=text_mask,
            speech_seg_pos=speech_seg_pos,
            text_seg_pos=text_seg_pos,
            span_bdy=span_bdy,
            use_teacher_forcing=use_teacher_forcing)


class ErnieSATInference(nn.Layer):
    def __init__(self, normalizer, model):
        super().__init__()
        self.normalizer = normalizer
        self.acoustic_model = model

    def forward(
            self,
            speech: paddle.Tensor,
            text: paddle.Tensor,
            masked_pos: paddle.Tensor,
            speech_mask: paddle.Tensor,
            text_mask: paddle.Tensor,
            speech_seg_pos: paddle.Tensor,
            text_seg_pos: paddle.Tensor,
            span_bdy: List[int],
            use_teacher_forcing: bool=True, ):
        outs = self.acoustic_model.inference(
            speech=speech,
            text=text,
            masked_pos=masked_pos,
            speech_mask=speech_mask,
            text_mask=text_mask,
            speech_seg_pos=speech_seg_pos,
            text_seg_pos=text_seg_pos,
            span_bdy=span_bdy,
            use_teacher_forcing=use_teacher_forcing)

        normed_mel_pre, normed_mel_masked, normed_mel_post = outs
        logmel_pre = self.normalizer.inverse(normed_mel_pre)
        logmel_masked = self.normalizer.inverse(normed_mel_masked)
        logmel_post = self.normalizer.inverse(normed_mel_post)
        return logmel_pre, logmel_masked, logmel_post
