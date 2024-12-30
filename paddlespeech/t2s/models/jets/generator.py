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
"""Generator module in JETS.

This code is based on https://github.com/imdanboy/jets.

"""
import logging
import math
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import paddle
from paddle import nn
from typeguard import check_argument_types

from paddlespeech.t2s.models.hifigan import HiFiGANGenerator
from paddlespeech.t2s.models.jets.alignments import AlignmentModule
from paddlespeech.t2s.models.jets.alignments import average_by_duration
from paddlespeech.t2s.models.jets.alignments import viterbi_decode
from paddlespeech.t2s.models.jets.length_regulator import GaussianUpsampling
from paddlespeech.t2s.modules.nets_utils import get_random_segments
from paddlespeech.t2s.modules.nets_utils import initialize
from paddlespeech.t2s.modules.nets_utils import make_non_pad_mask
from paddlespeech.t2s.modules.nets_utils import make_pad_mask
from paddlespeech.t2s.modules.predictor.duration_predictor import DurationPredictor
from paddlespeech.t2s.modules.predictor.length_regulator import LengthRegulator
from paddlespeech.t2s.modules.predictor.variance_predictor import VariancePredictor
from paddlespeech.t2s.modules.style_encoder import StyleEncoder
from paddlespeech.t2s.modules.transformer.embedding import PositionalEncoding
from paddlespeech.t2s.modules.transformer.embedding import ScaledPositionalEncoding
from paddlespeech.t2s.modules.transformer.encoder import ConformerEncoder
from paddlespeech.t2s.modules.transformer.encoder import TransformerEncoder


class JETSGenerator(nn.Layer):
    """Generator module in JETS.
    """

    def __init__(
            self,
            idim: int,
            odim: int,
            adim: int=256,
            aheads: int=2,
            elayers: int=4,
            eunits: int=1024,
            dlayers: int=4,
            dunits: int=1024,
            positionwise_layer_type: str="conv1d",
            positionwise_conv_kernel_size: int=1,
            use_scaled_pos_enc: bool=True,
            use_batch_norm: bool=True,
            encoder_normalize_before: bool=True,
            decoder_normalize_before: bool=True,
            encoder_concat_after: bool=False,
            decoder_concat_after: bool=False,
            reduction_factor: int=1,
            encoder_type: str="transformer",
            decoder_type: str="transformer",
            transformer_enc_dropout_rate: float=0.1,
            transformer_enc_positional_dropout_rate: float=0.1,
            transformer_enc_attn_dropout_rate: float=0.1,
            transformer_dec_dropout_rate: float=0.1,
            transformer_dec_positional_dropout_rate: float=0.1,
            transformer_dec_attn_dropout_rate: float=0.1,
            transformer_activation_type: str="relu",
            # only for conformer
            conformer_rel_pos_type: str="legacy",
            conformer_pos_enc_layer_type: str="rel_pos",
            conformer_self_attn_layer_type: str="rel_selfattn",
            conformer_activation_type: str="swish",
            use_macaron_style_in_conformer: bool=True,
            use_cnn_in_conformer: bool=True,
            zero_triu: bool=False,
            conformer_enc_kernel_size: int=7,
            conformer_dec_kernel_size: int=31,
            # duration predictor
            duration_predictor_layers: int=2,
            duration_predictor_chans: int=384,
            duration_predictor_kernel_size: int=3,
            duration_predictor_dropout_rate: float=0.1,
            # energy predictor
            energy_predictor_layers: int=2,
            energy_predictor_chans: int=384,
            energy_predictor_kernel_size: int=3,
            energy_predictor_dropout: float=0.5,
            energy_embed_kernel_size: int=9,
            energy_embed_dropout: float=0.5,
            stop_gradient_from_energy_predictor: bool=False,
            # pitch predictor
            pitch_predictor_layers: int=2,
            pitch_predictor_chans: int=384,
            pitch_predictor_kernel_size: int=3,
            pitch_predictor_dropout: float=0.5,
            pitch_embed_kernel_size: int=9,
            pitch_embed_dropout: float=0.5,
            stop_gradient_from_pitch_predictor: bool=False,
            # extra embedding related
            spks: Optional[int]=None,
            langs: Optional[int]=None,
            spk_embed_dim: Optional[int]=None,
            spk_embed_integration_type: str="add",
            use_gst: bool=False,
            gst_tokens: int=10,
            gst_heads: int=4,
            gst_conv_layers: int=6,
            gst_conv_chans_list: Sequence[int]=(32, 32, 64, 64, 128, 128),
            gst_conv_kernel_size: int=3,
            gst_conv_stride: int=2,
            gst_gru_layers: int=1,
            gst_gru_units: int=128,
            # training related
            init_type: str="xavier_uniform",
            init_enc_alpha: float=1.0,
            init_dec_alpha: float=1.0,
            use_masking: bool=False,
            use_weighted_masking: bool=False,
            segment_size: int=64,
            # hifigan generator
            generator_out_channels: int=1,
            generator_channels: int=512,
            generator_global_channels: int=-1,
            generator_kernel_size: int=7,
            generator_upsample_scales: List[int]=[8, 8, 2, 2],
            generator_upsample_kernel_sizes: List[int]=[16, 16, 4, 4],
            generator_resblock_kernel_sizes: List[int]=[3, 7, 11],
            generator_resblock_dilations: List[List[int]]=[[1, 3, 5], [1, 3, 5],
                                                           [1, 3, 5]],
            generator_use_additional_convs: bool=True,
            generator_bias: bool=True,
            generator_nonlinear_activation: str="LeakyReLU",
            generator_nonlinear_activation_params: Dict[
                str, Any]={"negative_slope": 0.1},
            generator_use_weight_norm: bool=True, ):
        """Initialize JETS generator module.

        Args:
            idim (int): 
                Dimension of the inputs.
            odim (int): 
                Dimension of the outputs.
            adim (int): 
                Attention dimension.
            aheads (int): 
                Number of attention heads.
            elayers (int): 
                Number of encoder layers.
            eunits (int): 
                Number of encoder hidden units.
            dlayers (int): 
                Number of decoder layers.
            dunits (int): 
                Number of decoder hidden units.
            use_scaled_pos_enc (bool): 
                Whether to use trainable scaled pos encoding.
            use_batch_norm (bool): 
                Whether to use batch normalization in encoder prenet.
            encoder_normalize_before (bool): 
                Whether to apply layernorm layer before encoder block.
            decoder_normalize_before (bool): 
                Whether to apply layernorm layer before decoder block.
            encoder_concat_after (bool): 
                Whether to concatenate attention layer's input and output in encoder.
            decoder_concat_after (bool): 
                Whether to concatenate attention layer's input and output in decoder.
            reduction_factor (int): 
                Reduction factor.
            encoder_type (str): 
                Encoder type ("transformer" or "conformer").
            decoder_type (str): 
                Decoder type ("transformer" or "conformer").
            transformer_enc_dropout_rate (float): 
                Dropout rate in encoder except attention and positional encoding.
            transformer_enc_positional_dropout_rate (float): 
                Dropout rate after encoder positional encoding.
            transformer_enc_attn_dropout_rate (float): 
                Dropout rate in encoder self-attention module.
            transformer_dec_dropout_rate (float): 
                Dropout rate in decoder except attention & positional encoding.
            transformer_dec_positional_dropout_rate (float): 
                Dropout rate after decoder positional encoding.
            transformer_dec_attn_dropout_rate (float): 
                Dropout rate in decoder self-attention module.
            conformer_rel_pos_type (str): 
                Relative pos encoding type in conformer.
            conformer_pos_enc_layer_type (str): 
                Pos encoding layer type in conformer.
            conformer_self_attn_layer_type (str): 
                Self-attention layer type in conformer
            conformer_activation_type (str): 
                Activation function type in conformer.
            use_macaron_style_in_conformer: 
                Whether to use macaron style FFN.
            use_cnn_in_conformer: 
                Whether to use CNN in conformer.
            zero_triu: 
                Whether to use zero triu in relative self-attention module.
            conformer_enc_kernel_size: 
                Kernel size of encoder conformer.
            conformer_dec_kernel_size: 
                Kernel size of decoder conformer.
            duration_predictor_layers (int): 
                Number of duration predictor layers.
            duration_predictor_chans (int): 
                Number of duration predictor channels.
            duration_predictor_kernel_size (int): 
                Kernel size of duration predictor.
            duration_predictor_dropout_rate (float): 
                Dropout rate in duration predictor.
            pitch_predictor_layers (int): 
                Number of pitch predictor layers.
            pitch_predictor_chans (int): 
                Number of pitch predictor channels.
            pitch_predictor_kernel_size (int): 
                Kernel size of pitch predictor.
            pitch_predictor_dropout_rate (float): 
                Dropout rate in pitch predictor.
            pitch_embed_kernel_size (float): 
                Kernel size of pitch embedding.
            pitch_embed_dropout_rate (float): 
                Dropout rate for pitch embedding.
            stop_gradient_from_pitch_predictor: 
                Whether to stop gradient from pitch predictor to encoder.
            energy_predictor_layers (int): 
                Number of energy predictor layers.
            energy_predictor_chans (int): 
                Number of energy predictor channels.
            energy_predictor_kernel_size (int): 
                Kernel size of energy predictor.
            energy_predictor_dropout_rate (float): 
                Dropout rate in energy predictor.
            energy_embed_kernel_size (float): 
                Kernel size of energy embedding.
            energy_embed_dropout_rate (float): 
                Dropout rate for energy embedding.
            stop_gradient_from_energy_predictor: 
                Whether to stop gradient from energy predictor to encoder.
            spks (Optional[int]): 
                Number of speakers. If set to > 1, assume that the sids will be provided as the input and use sid embedding layer.
            langs (Optional[int]): 
                Number of languages. If set to > 1, assume that the lids will be provided as the input and use sid embedding layer.
            spk_embed_dim (Optional[int]): 
                Speaker embedding dimension. If set to > 0, assume that spembs will be provided as the input.
            spk_embed_integration_type: 
                How to integrate speaker embedding.
            use_gst (str): 
                Whether to use global style token.
            gst_tokens (int): 
                The number of GST embeddings.
            gst_heads (int): 
                The number of heads in GST multihead attention.
            gst_conv_layers (int): 
                The number of conv layers in GST.
            gst_conv_chans_list: (Sequence[int]):
                List of the number of channels of conv layers in GST.
            gst_conv_kernel_size (int): 
                Kernel size of conv layers in GST.
            gst_conv_stride (int): 
                Stride size of conv layers in GST.
            gst_gru_layers (int): 
                The number of GRU layers in GST.
            gst_gru_units (int): 
                The number of GRU units in GST.
            init_type (str): 
                How to initialize transformer parameters.
            init_enc_alpha (float): 
                Initial value of alpha in scaled pos encoding of the encoder.
            init_dec_alpha (float): 
                Initial value of alpha in scaled pos encoding of the decoder.
            use_masking (bool): 
                Whether to apply masking for padded part in loss calculation.
            use_weighted_masking (bool): 
                Whether to apply weighted masking in loss calculation.
            segment_size (int): 
                Segment size for random windowed discriminator
            generator_out_channels (int): 
                Number of output channels.
            generator_channels (int): 
                Number of hidden representation channels.
            generator_global_channels (int): 
                Number of global conditioning channels.
            generator_kernel_size (int): 
                Kernel size of initial and final conv layer.
            generator_upsample_scales (List[int]): 
                List of upsampling scales.
            generator_upsample_kernel_sizes (List[int]): 
                List of kernel sizes for upsample layers.
            generator_resblock_kernel_sizes (List[int]): 
                List of kernel sizes for residual blocks.
            generator_resblock_dilations (List[List[int]]): 
                List of list of dilations for residual blocks.
            generator_use_additional_convs (bool): 
                Whether to use additional conv layers in residual blocks.
            generator_bias (bool): 
                Whether to add bias parameter in convolution layers.
            generator_nonlinear_activation (str): 
                Activation function module name.
            generator_nonlinear_activation_params (Dict[str, Any]): 
                Hyperparameters for activation function.
            generator_use_weight_norm (bool): 
                Whether to use weight norm. If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()
        self.segment_size = segment_size
        self.upsample_factor = int(np.prod(generator_upsample_scales))
        self.idim = idim
        self.odim = odim
        self.reduction_factor = reduction_factor
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.stop_gradient_from_pitch_predictor = stop_gradient_from_pitch_predictor
        self.stop_gradient_from_energy_predictor = stop_gradient_from_energy_predictor
        self.use_scaled_pos_enc = use_scaled_pos_enc
        self.use_gst = use_gst

        # use idx 0 as padding idx
        self.padding_idx = 0

        # get positional encoding layer type
        transformer_pos_enc_layer_type = "scaled_abs_pos" if self.use_scaled_pos_enc else "abs_pos"

        # check relative positional encoding compatibility
        if "conformer" in [encoder_type, decoder_type]:
            if conformer_rel_pos_type == "legacy":
                if conformer_pos_enc_layer_type == "rel_pos":
                    conformer_pos_enc_layer_type = "legacy_rel_pos"
                    logging.warning(
                        "Fallback to conformer_pos_enc_layer_type = 'legacy_rel_pos' "
                        "due to the compatibility. If you want to use the new one, "
                        "please use conformer_pos_enc_layer_type = 'latest'.")
                if conformer_self_attn_layer_type == "rel_selfattn":
                    conformer_self_attn_layer_type = "legacy_rel_selfattn"
                    logging.warning(
                        "Fallback to "
                        "conformer_self_attn_layer_type = 'legacy_rel_selfattn' "
                        "due to the compatibility. If you want to use the new one, "
                        "please use conformer_pos_enc_layer_type = 'latest'.")
            elif conformer_rel_pos_type == "latest":
                assert conformer_pos_enc_layer_type != "legacy_rel_pos"
                assert conformer_self_attn_layer_type != "legacy_rel_selfattn"
            else:
                raise ValueError(
                    f"Unknown rel_pos_type: {conformer_rel_pos_type}")

        # define encoder
        encoder_input_layer = nn.Embedding(
            num_embeddings=idim,
            embedding_dim=adim,
            padding_idx=self.padding_idx)
        if encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                idim=idim,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=eunits,
                num_blocks=elayers,
                input_layer=encoder_input_layer,
                dropout_rate=transformer_enc_dropout_rate,
                positional_dropout_rate=transformer_enc_positional_dropout_rate,
                attention_dropout_rate=transformer_enc_attn_dropout_rate,
                pos_enc_layer_type=transformer_pos_enc_layer_type,
                normalize_before=encoder_normalize_before,
                concat_after=encoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                activation_type=transformer_activation_type)
        elif encoder_type == "conformer":
            self.encoder = ConformerEncoder(
                idim=idim,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=eunits,
                num_blocks=elayers,
                input_layer=encoder_input_layer,
                dropout_rate=transformer_enc_dropout_rate,
                positional_dropout_rate=transformer_enc_positional_dropout_rate,
                attention_dropout_rate=transformer_enc_attn_dropout_rate,
                normalize_before=encoder_normalize_before,
                concat_after=encoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                macaron_style=use_macaron_style_in_conformer,
                pos_enc_layer_type=conformer_pos_enc_layer_type,
                selfattention_layer_type=conformer_self_attn_layer_type,
                activation_type=conformer_activation_type,
                use_cnn_module=use_cnn_in_conformer,
                cnn_module_kernel=conformer_enc_kernel_size,
                zero_triu=zero_triu, )
        else:
            raise ValueError(f"{encoder_type} is not supported.")

        # define GST
        if self.use_gst:
            self.gst = StyleEncoder(
                idim=odim,  # the input is mel-spectrogram
                gst_tokens=gst_tokens,
                gst_token_dim=adim,
                gst_heads=gst_heads,
                conv_layers=gst_conv_layers,
                conv_chans_list=gst_conv_chans_list,
                conv_kernel_size=gst_conv_kernel_size,
                conv_stride=gst_conv_stride,
                gru_layers=gst_gru_layers,
                gru_units=gst_gru_units, )

        # define spk and lang embedding
        self.spks = None
        if spks is not None and spks > 1:
            self.spks = spks
            self.sid_emb = nn.Embedding(spks, adim)
        self.langs = None
        if langs is not None and langs > 1:
            self.langs = langs
            self.lid_emb = nn.Embedding(langs, adim)

        # define additional projection for speaker embedding
        self.spk_embed_dim = None
        if spk_embed_dim is not None and spk_embed_dim > 0:
            self.spk_embed_dim = spk_embed_dim
            self.spk_embed_integration_type = spk_embed_integration_type
        if self.spk_embed_dim is not None:
            if self.spk_embed_integration_type == "add":
                self.projection = nn.Linear(self.spk_embed_dim, adim)
            else:
                self.projection = nn.Linear(adim + self.spk_embed_dim, adim)

        # define duration predictor
        self.duration_predictor = DurationPredictor(
            idim=adim,
            n_layers=duration_predictor_layers,
            n_chans=duration_predictor_chans,
            kernel_size=duration_predictor_kernel_size,
            dropout_rate=duration_predictor_dropout_rate, )

        # define pitch predictor
        self.pitch_predictor = VariancePredictor(
            idim=adim,
            n_layers=pitch_predictor_layers,
            n_chans=pitch_predictor_chans,
            kernel_size=pitch_predictor_kernel_size,
            dropout_rate=pitch_predictor_dropout, )
        # NOTE(kan-bayashi): We use continuous pitch + FastPitch style avg
        self.pitch_embed = nn.Sequential(
            nn.Conv1D(
                in_channels=1,
                out_channels=adim,
                kernel_size=pitch_embed_kernel_size,
                padding=(pitch_embed_kernel_size - 1) // 2, ),
            nn.Dropout(pitch_embed_dropout), )

        # define energy predictor
        self.energy_predictor = VariancePredictor(
            idim=adim,
            n_layers=energy_predictor_layers,
            n_chans=energy_predictor_chans,
            kernel_size=energy_predictor_kernel_size,
            dropout_rate=energy_predictor_dropout, )
        # NOTE(kan-bayashi): We use continuous enegy + FastPitch style avg
        self.energy_embed = nn.Sequential(
            nn.Conv1D(
                in_channels=1,
                out_channels=adim,
                kernel_size=energy_embed_kernel_size,
                padding=(energy_embed_kernel_size - 1) // 2, ),
            nn.Dropout(energy_embed_dropout), )

        # define length regulator
        self.length_regulator = GaussianUpsampling()

        # define decoder
        # NOTE: we use encoder as decoder
        # because fastspeech's decoder is the same as encoder
        if decoder_type == "transformer":
            self.decoder = TransformerEncoder(
                idim=0,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=dunits,
                num_blocks=dlayers,
                # in decoder, don't need layer before pos_enc_class (we use embedding here in encoder)
                input_layer=None,
                dropout_rate=transformer_dec_dropout_rate,
                positional_dropout_rate=transformer_dec_positional_dropout_rate,
                attention_dropout_rate=transformer_dec_attn_dropout_rate,
                pos_enc_layer_type=transformer_pos_enc_layer_type,
                normalize_before=decoder_normalize_before,
                concat_after=decoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                activation_type=conformer_activation_type, )

        elif decoder_type == "conformer":
            self.decoder = ConformerEncoder(
                idim=0,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=dunits,
                num_blocks=dlayers,
                input_layer=None,
                dropout_rate=transformer_dec_dropout_rate,
                positional_dropout_rate=transformer_dec_positional_dropout_rate,
                attention_dropout_rate=transformer_dec_attn_dropout_rate,
                normalize_before=decoder_normalize_before,
                concat_after=decoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                macaron_style=use_macaron_style_in_conformer,
                pos_enc_layer_type=conformer_pos_enc_layer_type,
                selfattention_layer_type=conformer_self_attn_layer_type,
                activation_type=conformer_activation_type,
                use_cnn_module=use_cnn_in_conformer,
                cnn_module_kernel=conformer_dec_kernel_size, )
        else:
            raise ValueError(f"{decoder_type} is not supported.")

        self.generator = HiFiGANGenerator(
            in_channels=adim,
            out_channels=generator_out_channels,
            channels=generator_channels,
            global_channels=generator_global_channels,
            kernel_size=generator_kernel_size,
            upsample_scales=generator_upsample_scales,
            upsample_kernel_sizes=generator_upsample_kernel_sizes,
            resblock_kernel_sizes=generator_resblock_kernel_sizes,
            resblock_dilations=generator_resblock_dilations,
            use_additional_convs=generator_use_additional_convs,
            bias=generator_bias,
            nonlinear_activation=generator_nonlinear_activation,
            nonlinear_activation_params=generator_nonlinear_activation_params,
            use_weight_norm=generator_use_weight_norm, )

        self.alignment_module = AlignmentModule(adim, odim)

        # initialize parameters
        self._reset_parameters(
            init_type=init_type,
            init_enc_alpha=init_enc_alpha,
            init_dec_alpha=init_dec_alpha, )

    def forward(
            self,
            text: paddle.Tensor,
            text_lengths: paddle.Tensor,
            feats: paddle.Tensor,
            feats_lengths: paddle.Tensor,
            durations: paddle.Tensor,
            durations_lengths: paddle.Tensor,
            pitch: paddle.Tensor,
            energy: paddle.Tensor,
            sids: Optional[paddle.Tensor]=None,
            spembs: Optional[paddle.Tensor]=None,
            lids: Optional[paddle.Tensor]=None,
            use_alignment_module: bool=False,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor,
               paddle.Tensor, paddle.Tensor,
               Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor,
                     paddle.Tensor, paddle.Tensor, ], ]:
        """Calculate forward propagation.
        Args:
            text (Tensor):
                Text index tensor (B, T_text).
            text_lengths (Tensor):
                Text length tensor (B,).
            feats (Tensor):
                Feature tensor (B, aux_channels, T_feats).
            feats_lengths (Tensor):
                Feature length tensor (B,).
            pitch (Tensor): 
                Batch of padded token-averaged pitch (B, T_text, 1).
            energy (Tensor):
                Batch of padded token-averaged energy (B, T_text, 1).
            sids (Optional[Tensor]):
                Speaker index tensor (B,) or (B, 1).
            spembs (Optional[Tensor]):
                Speaker embedding tensor (B, spk_embed_dim).
            lids (Optional[Tensor]):
                Language index tensor (B,) or (B, 1).
            use_alignment_module (bool):
                Whether to use alignment module.
                
        Returns:
            Tensor: 
                Waveform tensor (B, 1, segment_size * upsample_factor).
            Tensor: 
                binarization loss ()
            Tensor: 
                log probability attention matrix (B,T_feats,T_text)
            Tensor: 
                Segments start index tensor (B,).
            Tensor: 
                predicted duration (B,T_text)
            Tensor: 
                ground-truth duration obtained from an alignment module (B,T_text)
            Tensor: 
                predicted pitch (B,T_text,1)
            Tensor: 
                ground-truth averaged pitch (B,T_text,1)
            Tensor: 
                predicted energy (B,T_text,1)
            Tensor: 
                ground-truth averaged energy (B,T_text,1)
        """
        if use_alignment_module:
            text = text[:, :text_lengths.max()]  # for data-parallel
            feats = feats[:, :feats_lengths.max()]  # for data-parallel
            pitch = pitch[:, :durations_lengths.max()]  # for data-parallel
            energy = energy[:, :durations_lengths.max()]  # for data-parallel
        else:
            text = text[:, :text_lengths.max()]  # for data-parallel
            feats = feats[:, :feats_lengths.max()]  # for data-parallel
            pitch = pitch[:, :feats_lengths.max()]  # for data-parallel
            energy = energy[:, :feats_lengths.max()]  # for data-parallel

        # forward encoder
        x_masks = self._source_mask(text_lengths)
        hs, _ = self.encoder(text, x_masks)  # (B, T_text, adim)

        # integrate with GST
        if self.use_gst:
            style_embs = self.gst(ys)
            hs = hs + style_embs.unsqueeze(1)

        # integrate with SID and LID embeddings
        if self.spks is not None:
            sid_embs = self.sid_emb(sids.reshape([-1]))
            hs = hs + sid_embs.unsqueeze(1)
        if self.langs is not None:
            lid_embs = self.lid_emb(lids.reshape([-1]))
            hs = hs + lid_embs.unsqueeze(1)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)

        # forward alignment module and obtain duration, averaged pitch, energy
        h_masks = make_pad_mask(text_lengths)
        if use_alignment_module:
            log_p_attn = self.alignment_module(hs, feats, h_masks)
            ds, bin_loss = viterbi_decode(log_p_attn, text_lengths,
                                          feats_lengths)
            ps = average_by_duration(ds,
                                     pitch.squeeze(-1), text_lengths,
                                     feats_lengths).unsqueeze(-1)
            es = average_by_duration(ds,
                                     energy.squeeze(-1), text_lengths,
                                     feats_lengths).unsqueeze(-1)
        else:
            ds = durations
            ps = pitch
            es = energy
            log_p_attn = attn = bin_loss = None

        # forward duration predictor and variance predictors
        if self.stop_gradient_from_pitch_predictor:
            p_outs = self.pitch_predictor(hs.detach(), h_masks.unsqueeze(-1))
        else:
            p_outs = self.pitch_predictor(hs, h_masks.unsqueeze(-1))
        if self.stop_gradient_from_energy_predictor:
            e_outs = self.energy_predictor(hs.detach(), h_masks.unsqueeze(-1))
        else:
            e_outs = self.energy_predictor(hs, h_masks.unsqueeze(-1))

        d_outs = self.duration_predictor(hs, h_masks)

        # use groundtruth in training
        p_embs = self.pitch_embed(ps.transpose([0, 2, 1])).transpose([0, 2, 1])
        e_embs = self.energy_embed(es.transpose([0, 2, 1])).transpose([0, 2, 1])
        hs = hs + e_embs + p_embs

        # upsampling
        h_masks = make_non_pad_mask(feats_lengths)
        # d_masks = make_non_pad_mask(text_lengths).to(ds.device)
        d_masks = make_non_pad_mask(text_lengths)
        hs = self.length_regulator(hs, ds, h_masks,
                                   d_masks)  # (B, T_feats, adim)

        # forward decoder
        h_masks = self._source_mask(feats_lengths)
        zs, _ = self.decoder(hs, h_masks)  # (B, T_feats, adim)

        # get random segments
        z_segments, z_start_idxs = get_random_segments(
            zs.transpose([0, 2, 1]),
            feats_lengths,
            self.segment_size, )
        # forward generator
        wav = self.generator(z_segments)
        if use_alignment_module:
            return wav, bin_loss, log_p_attn, z_start_idxs, d_outs, ds, p_outs, ps, e_outs, es
        else:
            return wav, None, None, z_start_idxs, d_outs, ds, p_outs, ps, e_outs, es

    def inference(
            self,
            text: paddle.Tensor,
            text_lengths: paddle.Tensor,
            feats: Optional[paddle.Tensor]=None,
            feats_lengths: Optional[paddle.Tensor]=None,
            pitch: Optional[paddle.Tensor]=None,
            energy: Optional[paddle.Tensor]=None,
            sids: Optional[paddle.Tensor]=None,
            spembs: Optional[paddle.Tensor]=None,
            lids: Optional[paddle.Tensor]=None,
            use_alignment_module: bool=False,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Run inference.

        Args:
            text (Tensor): Input text index tensor (B, T_text,).
            text_lengths (Tensor): Text length tensor (B,).
            feats (Tensor): Feature tensor (B, T_feats, aux_channels).
            feats_lengths (Tensor): Feature length tensor (B,).
            pitch (Tensor): Pitch tensor (B, T_feats, 1)
            energy (Tensor): Energy tensor (B, T_feats, 1)
            sids (Optional[Tensor]): Speaker index tensor (B,) or (B, 1).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, spk_embed_dim).
            lids (Optional[Tensor]): Language index tensor (B,) or (B, 1).
            use_alignment_module (bool): Whether to use alignment module.

        Returns:
            Tensor: Generated waveform tensor (B, T_wav).
            Tensor: Duration tensor (B, T_text).

        """
        # forward encoder
        x_masks = self._source_mask(text_lengths)
        hs, _ = self.encoder(text, x_masks)  # (B, T_text, adim)

        # integrate with GST
        if self.use_gst:
            style_embs = self.gst(ys)
            hs = hs + style_embs.unsqueeze(1)

        # integrate with SID and LID embeddings
        if self.spks is not None:
            sid_embs = self.sid_emb(sids.reshape([-1]))
            hs = hs + sid_embs.unsqueeze(1)
        if self.langs is not None:
            lid_embs = self.lid_emb(lids.reshape([-1]))
            hs = hs + lid_embs.unsqueeze(1)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)

        h_masks = make_pad_mask(text_lengths)
        if use_alignment_module:
            # forward alignment module and obtain duration, averaged pitch, energy
            log_p_attn, attn = self.alignment_module(hs, feats, h_masks)
            d_outs, _ = viterbi_decode(log_p_attn, text_lengths, feats_lengths)
            p_outs = average_by_duration(d_outs,
                                         pitch.squeeze(-1), text_lengths,
                                         feats_lengths).unsqueeze(-1)
            e_outs = average_by_duration(d_outs,
                                         energy.squeeze(-1), text_lengths,
                                         feats_lengths).unsqueeze(-1)
        else:
            # forward duration predictor and variance predictors
            p_outs = self.pitch_predictor(hs, h_masks.unsqueeze(-1))
            e_outs = self.energy_predictor(hs, h_masks.unsqueeze(-1))
            d_outs = self.duration_predictor.inference(hs, h_masks)

        p_embs = self.pitch_embed(p_outs.transpose([0, 2, 1])).transpose(
            [0, 2, 1])
        e_embs = self.energy_embed(e_outs.transpose([0, 2, 1])).transpose(
            [0, 2, 1])
        hs = hs + e_embs + p_embs

        # upsampling
        if feats_lengths is not None:
            h_masks = make_non_pad_mask(feats_lengths)
        else:
            h_masks = None
        d_masks = make_non_pad_mask(text_lengths)
        hs = self.length_regulator(hs, d_outs, h_masks,
                                   d_masks)  # (B, T_feats, adim)

        # forward decoder
        if feats_lengths is not None:
            h_masks = self._source_mask(feats_lengths)
        else:
            h_masks = None
        zs, _ = self.decoder(hs, h_masks)  # (B, T_feats, adim)

        # forward generator
        wav = self.generator(zs.transpose([0, 2, 1]))

        return wav.squeeze(1), d_outs

    def _integrate_with_spk_embed(self,
                                  hs: paddle.Tensor,
                                  spembs: paddle.Tensor) -> paddle.Tensor:
        """Integrate speaker embedding with hidden states.

        Args:
            hs (Tensor): Batch of hidden state sequences (B, T_text, adim).
            spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, T_text, adim).

        """
        if self.spk_embed_integration_type == "add":
            # apply projection and then add to hidden states
            spembs = self.projection(F.normalize(spembs))
            hs = hs + spembs.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # concat hidden states with spk embeds and then apply projection
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.shape[1],
                                                             -1)
            hs = self.projection(paddle.concat([hs, spembs], axis=-1))
        else:
            raise NotImplementedError("support only add or concat.")

        return hs

    def _generate_path(self, dur: paddle.Tensor,
                       mask: paddle.Tensor) -> paddle.Tensor:
        """Generate path a.k.a. monotonic attention.
        Args:
            dur (Tensor):
                Duration tensor (B, 1, T_text).
            mask (Tensor):
                Attention mask tensor (B, 1, T_feats, T_text).
        Returns:
            Tensor:
                Path tensor (B, 1, T_feats, T_text).
        """
        b, _, t_y, t_x = paddle.shape(mask)
        cum_dur = paddle.cumsum(dur, -1)
        cum_dur_flat = paddle.reshape(cum_dur, [b * t_x])

        path = paddle.arange(t_y, dtype=dur.dtype)
        path = path.unsqueeze(0) < cum_dur_flat.unsqueeze(1)
        path = paddle.reshape(path, [b, t_x, t_y])
        '''
        path will be like (t_x = 3, t_y = 5):
        [[[1., 1., 0., 0., 0.],      [[[1., 1., 0., 0., 0.],
          [1., 1., 1., 1., 0.],  -->   [0., 0., 1., 1., 0.],
          [1., 1., 1., 1., 1.]]]       [0., 0., 0., 0., 1.]]]
        '''

        path = paddle.cast(path, dtype='float32')
        pad_tmp = self.pad1d(path)[:, :-1]
        path = path - pad_tmp
        return path.unsqueeze(1).transpose([0, 1, 3, 2]) * mask

    def _source_mask(self, ilens: paddle.Tensor) -> paddle.Tensor:
        """Make masks for self-attention.

        Args:
            ilens (LongTensor): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for self-attention.
                dtype=paddle.uint8 

        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 0, 0]]], dtype=torch.uint8)

        """
        x_masks = paddle.to_tensor(make_non_pad_mask(ilens))
        return x_masks.unsqueeze(-2)

    def _reset_parameters(self,
                          init_type: str,
                          init_enc_alpha: float,
                          init_dec_alpha: float):
        # initialize parameters
        initialize(self, init_type)

        # initialize alpha in scaled positional encoding
        if self.encoder_type == "transformer" and self.use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = paddle.to_tensor(init_enc_alpha)
        if self.decoder_type == "transformer" and self.use_scaled_pos_enc:
            self.decoder.embed[-1].alpha.data = paddle.to_tensor(init_dec_alpha)
