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
# Modified from espnet(https://github.com/espnet/espnet)
"""Fastspeech2 related modules for paddle"""
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn
from typeguard import check_argument_types

from paddlespeech.t2s.modules.adversarial_loss.gradient_reversal import GradientReversalLayer
from paddlespeech.t2s.modules.adversarial_loss.speaker_classifier import SpeakerClassifier
from paddlespeech.t2s.modules.nets_utils import initialize
from paddlespeech.t2s.modules.nets_utils import make_non_pad_mask
from paddlespeech.t2s.modules.nets_utils import make_pad_mask
from paddlespeech.t2s.modules.predictor.duration_predictor import DurationPredictor
from paddlespeech.t2s.modules.predictor.duration_predictor import DurationPredictorLoss
from paddlespeech.t2s.modules.predictor.length_regulator import LengthRegulator
from paddlespeech.t2s.modules.predictor.variance_predictor import VariancePredictor
from paddlespeech.t2s.modules.tacotron2.decoder import Postnet
from paddlespeech.t2s.modules.transformer.encoder import CNNDecoder
from paddlespeech.t2s.modules.transformer.encoder import CNNPostnet
from paddlespeech.t2s.modules.transformer.encoder import ConformerEncoder
from paddlespeech.t2s.modules.transformer.encoder import TransformerEncoder


class FastSpeech2(nn.Layer):
    """FastSpeech2 module.
    
    This is a module of FastSpeech2 described in `FastSpeech 2: Fast and
    High-Quality End-to-End Text to Speech`_. Instead of quantized pitch and
    energy, we use token-averaged value introduced in `FastPitch: Parallel
    Text-to-speech with Pitch Prediction`_.
    
    .. _`FastSpeech 2: Fast and High-Quality End-to-End Text to Speech`:
        https://arxiv.org/abs/2006.04558
    .. _`FastPitch: Parallel Text-to-speech with Pitch Prediction`:
        https://arxiv.org/abs/2006.06873

    Args:
    
    Returns:

    """

    def __init__(
            self,
            # network structure related
            idim: int,
            odim: int,
            adim: int=384,
            aheads: int=4,
            elayers: int=6,
            eunits: int=1536,
            dlayers: int=6,
            dunits: int=1536,
            postnet_layers: int=5,
            postnet_chans: int=512,
            postnet_filts: int=5,
            postnet_dropout_rate: float=0.5,
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
            # for transformer
            transformer_enc_dropout_rate: float=0.1,
            transformer_enc_positional_dropout_rate: float=0.1,
            transformer_enc_attn_dropout_rate: float=0.1,
            transformer_dec_dropout_rate: float=0.1,
            transformer_dec_positional_dropout_rate: float=0.1,
            transformer_dec_attn_dropout_rate: float=0.1,
            transformer_activation_type: str="relu",
            # for conformer
            conformer_pos_enc_layer_type: str="rel_pos",
            conformer_self_attn_layer_type: str="rel_selfattn",
            conformer_activation_type: str="swish",
            use_macaron_style_in_conformer: bool=True,
            use_cnn_in_conformer: bool=True,
            zero_triu: bool=False,
            conformer_enc_kernel_size: int=7,
            conformer_dec_kernel_size: int=31,
            # for CNN Decoder
            cnn_dec_dropout_rate: float=0.2,
            cnn_postnet_dropout_rate: float=0.2,
            cnn_postnet_resblock_kernel_sizes: List[int]=[256, 256],
            cnn_postnet_kernel_size: int=5,
            cnn_decoder_embedding_dim: int=256,
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
            # spk emb
            spk_num: int=None,
            spk_embed_dim: int=None,
            spk_embed_integration_type: str="add",
            # tone emb
            tone_num: int=None,
            tone_embed_dim: int=None,
            tone_embed_integration_type: str="add",
            # training related
            init_type: str="xavier_uniform",
            init_enc_alpha: float=1.0,
            init_dec_alpha: float=1.0,
            # speaker classifier
            enable_speaker_classifier: bool=False,
            hidden_sc_dim: int=256, ):
        """Initialize FastSpeech2 module.
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
            postnet_layers (int): 
                Number of postnet layers.
            postnet_chans (int): 
                Number of postnet channels.
            postnet_filts (int): 
                Kernel size of postnet.
            postnet_dropout_rate (float): 
                Dropout rate in postnet.
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
                Whether to concatenate attention layer's input  and output in decoder.
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
            transformer_activation_type (str): 
                Activation function type in transformer.
            conformer_pos_enc_layer_type (str): 
                Pos encoding layer type in conformer.
            conformer_self_attn_layer_type (str): 
                Self-attention layer type in conformer
            conformer_activation_type (str): 
                Activation function type in conformer.
            use_macaron_style_in_conformer (bool): 
                Whether to use macaron style FFN.
            use_cnn_in_conformer (bool): 
                Whether to use CNN in conformer.
            zero_triu (bool): 
                Whether to use zero triu in relative self-attention module.
            conformer_enc_kernel_size (int): 
                Kernel size of encoder conformer.
            conformer_dec_kernel_size (int): 
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
            stop_gradient_from_pitch_predictor (bool): 
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
            stop_gradient_from_energy_predictor (bool): 
                Whether to stop gradient from energy predictor to encoder.
            spk_num (Optional[int]): 
                Number of speakers. If not None, assume that the spk_embed_dim is not None,
                spk_ids will be provided as the input and use spk_embedding_table.
            spk_embed_dim (Optional[int]): 
                Speaker embedding dimension. If not None, 
                assume that spk_emb will be provided as the input or spk_num is not None.
            spk_embed_integration_type (str): 
                How to integrate speaker embedding.
            tone_num (Optional[int]): 
                Number of tones. If not None, assume that the
                tone_ids will be provided as the input and use tone_embedding_table.
            tone_embed_dim (Optional[int]):
                Tone embedding dimension. If not None, assume that tone_num is not None.
            tone_embed_integration_type (str): 
                How to integrate tone embedding.
            init_type (str): 
                How to initialize transformer parameters.
            init_enc_alpha (float): 
                Initial value of alpha in scaled pos encoding of the encoder.
            init_dec_alpha (float): 
                Initial value of alpha in scaled pos encoding of the decoder.
            enable_speaker_classifier (bool):
                Whether to use speaker classifier module
            hidden_sc_dim (int):
                The hidden layer dim of speaker classifier
    
        """
        assert check_argument_types()
        super().__init__()

        # store hyperparameters
        self.odim = odim
        self.reduction_factor = reduction_factor
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.stop_gradient_from_pitch_predictor = stop_gradient_from_pitch_predictor
        self.stop_gradient_from_energy_predictor = stop_gradient_from_energy_predictor
        self.use_scaled_pos_enc = use_scaled_pos_enc
        self.hidden_sc_dim = hidden_sc_dim
        self.spk_num = spk_num
        self.enable_speaker_classifier = enable_speaker_classifier

        self.spk_embed_dim = spk_embed_dim
        if self.spk_embed_dim is not None:
            self.spk_embed_integration_type = spk_embed_integration_type

        self.tone_embed_dim = tone_embed_dim
        if self.tone_embed_dim is not None:
            self.tone_embed_integration_type = tone_embed_integration_type

        # use idx 0 as padding idx
        self.padding_idx = 0

        # initialize parameters
        initialize(self, init_type)

        if spk_num and self.spk_embed_dim:
            self.spk_embedding_table = nn.Embedding(
                num_embeddings=spk_num,
                embedding_dim=self.spk_embed_dim,
                padding_idx=self.padding_idx)

        if self.tone_embed_dim is not None:
            self.tone_embedding_table = nn.Embedding(
                num_embeddings=tone_num,
                embedding_dim=self.tone_embed_dim,
                padding_idx=self.padding_idx)

        # get positional encoding layer type
        transformer_pos_enc_layer_type = "scaled_abs_pos" if self.use_scaled_pos_enc else "abs_pos"

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

        # define additional projection for speaker embedding
        if self.spk_embed_dim is not None:
            if self.spk_embed_integration_type == "add":
                self.spk_projection = nn.Linear(self.spk_embed_dim, adim)
            else:
                self.spk_projection = nn.Linear(adim + self.spk_embed_dim, adim)

        # define additional projection for tone embedding
        if self.tone_embed_dim is not None:
            if self.tone_embed_integration_type == "add":
                self.tone_projection = nn.Linear(self.tone_embed_dim, adim)
            else:
                self.tone_projection = nn.Linear(adim + self.tone_embed_dim,
                                                 adim)

        if self.spk_num and self.enable_speaker_classifier:
            # set lambda = 1
            self.grad_reverse = GradientReversalLayer(1)
            self.speaker_classifier = SpeakerClassifier(
                idim=adim, hidden_sc_dim=self.hidden_sc_dim, spk_num=spk_num)

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
        #  We use continuous pitch + FastPitch style avg
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
        # We use continuous enegy + FastPitch style avg
        self.energy_embed = nn.Sequential(
            nn.Conv1D(
                in_channels=1,
                out_channels=adim,
                kernel_size=energy_embed_kernel_size,
                padding=(energy_embed_kernel_size - 1) // 2, ),
            nn.Dropout(energy_embed_dropout), )

        # define length regulator
        self.length_regulator = LengthRegulator()

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
        elif decoder_type == 'cnndecoder':
            self.decoder = CNNDecoder(
                emb_dim=adim,
                odim=odim,
                kernel_size=cnn_postnet_kernel_size,
                dropout_rate=cnn_dec_dropout_rate,
                resblock_kernel_sizes=cnn_postnet_resblock_kernel_sizes)
        else:
            raise ValueError(f"{decoder_type} is not supported.")

        # define final projection
        self.feat_out = nn.Linear(adim, odim * reduction_factor)

        # define postnet
        if decoder_type == 'cnndecoder':
            self.postnet = CNNPostnet(
                odim=odim,
                kernel_size=cnn_postnet_kernel_size,
                dropout_rate=cnn_postnet_dropout_rate,
                resblock_kernel_sizes=cnn_postnet_resblock_kernel_sizes)
        else:
            self.postnet = (None if postnet_layers == 0 else Postnet(
                idim=idim,
                odim=odim,
                n_layers=postnet_layers,
                n_chans=postnet_chans,
                n_filts=postnet_filts,
                use_batch_norm=use_batch_norm,
                dropout_rate=postnet_dropout_rate, ))

        nn.initializer.set_global_initializer(None)

        self._reset_parameters(
            init_enc_alpha=init_enc_alpha,
            init_dec_alpha=init_dec_alpha, )

    def forward(
            self,
            text: paddle.Tensor,
            text_lengths: paddle.Tensor,
            speech: paddle.Tensor,
            speech_lengths: paddle.Tensor,
            durations: paddle.Tensor,
            pitch: paddle.Tensor,
            energy: paddle.Tensor,
            tone_id: paddle.Tensor=None,
            spk_emb: paddle.Tensor=None,
            spk_id: paddle.Tensor=None
    ) -> Tuple[paddle.Tensor, Dict[str, paddle.Tensor], paddle.Tensor]:
        """Calculate forward propagation.

        Args:
            text(Tensor(int64)): 
                Batch of padded token ids (B, Tmax).
            text_lengths(Tensor(int64)): 
                Batch of lengths of each input (B,).
            speech(Tensor): 
                Batch of padded target features (B, Lmax, odim).
            speech_lengths(Tensor(int64)): 
                Batch of the lengths of each target (B,).
            durations(Tensor(int64)): 
                Batch of padded durations (B, Tmax).
            pitch(Tensor): 
                Batch of padded token-averaged pitch (B, Tmax, 1).
            energy(Tensor): 
                Batch of padded token-averaged energy (B, Tmax, 1).
            tone_id(Tensor, optional(int64)): 
                Batch of padded tone ids  (B, Tmax).
            spk_emb(Tensor, optional): 
                Batch of speaker embeddings (B, spk_embed_dim).
            spk_id(Tnesor, optional(int64)): 
                Batch of speaker ids (B,)

        Returns:

        """

        # input of embedding must be int64
        xs = paddle.cast(text, 'int64')
        ilens = paddle.cast(text_lengths, 'int64')
        ds = paddle.cast(durations, 'int64')
        olens = paddle.cast(speech_lengths, 'int64')
        ys = speech
        ps = pitch
        es = energy
        if spk_id is not None:
            spk_id = paddle.cast(spk_id, 'int64')
        if tone_id is not None:
            tone_id = paddle.cast(tone_id, 'int64')
        # forward propagation
        before_outs, after_outs, d_outs, p_outs, e_outs, spk_logits = self._forward(
            xs,
            ilens,
            olens,
            ds,
            ps,
            es,
            is_inference=False,
            spk_emb=spk_emb,
            spk_id=spk_id,
            tone_id=tone_id)
        # modify mod part of groundtruth
        if self.reduction_factor > 1:
            olens = olens - olens % self.reduction_factor
            max_olen = max(olens)
            ys = ys[:, :max_olen]

        return before_outs, after_outs, d_outs, p_outs, e_outs, ys, olens, spk_logits

    def _forward(self,
                 xs: paddle.Tensor,
                 ilens: paddle.Tensor,
                 olens: paddle.Tensor=None,
                 ds: paddle.Tensor=None,
                 ps: paddle.Tensor=None,
                 es: paddle.Tensor=None,
                 is_inference: bool=False,
                 return_after_enc=False,
                 alpha: float=1.0,
                 spk_emb=None,
                 spk_id=None,
                 tone_id=None) -> Sequence[paddle.Tensor]:
        # forward encoder
        x_masks = self._source_mask(ilens)
        # (B, Tmax, adim)
        hs, _ = self.encoder(xs, x_masks)

        if self.spk_num and self.enable_speaker_classifier and not is_inference:
            hs_for_spk_cls = self.grad_reverse(hs)
            spk_logits = self.speaker_classifier(hs_for_spk_cls, ilens)
        else:
            spk_logits = None

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            # spk_emb has a higher priority than spk_id
            if spk_emb is not None:
                hs = self._integrate_with_spk_embed(hs, spk_emb)
            elif spk_id is not None:
                spk_emb = self.spk_embedding_table(spk_id)
                hs = self._integrate_with_spk_embed(hs, spk_emb)

        # integrate tone embedding
        if self.tone_embed_dim is not None:
            if tone_id is not None:
                tone_embs = self.tone_embedding_table(tone_id)
                hs = self._integrate_with_tone_embed(hs, tone_embs)
        # forward duration predictor and variance predictors
        d_masks = make_pad_mask(ilens)

        if self.stop_gradient_from_pitch_predictor:
            p_outs = self.pitch_predictor(hs.detach(), d_masks.unsqueeze(-1))
        else:
            p_outs = self.pitch_predictor(hs, d_masks.unsqueeze(-1))
        if self.stop_gradient_from_energy_predictor:
            e_outs = self.energy_predictor(hs.detach(), d_masks.unsqueeze(-1))
        else:
            e_outs = self.energy_predictor(hs, d_masks.unsqueeze(-1))

        if is_inference:
            # (B, Tmax)
            if ds is not None:
                d_outs = ds
            else:
                d_outs = self.duration_predictor.inference(hs, d_masks)
            if ps is not None:
                p_outs = ps
            if es is not None:
                e_outs = es

            # use prediction in inference
            # (B, Tmax, 1)

            p_embs = self.pitch_embed(p_outs.transpose((0, 2, 1))).transpose(
                (0, 2, 1))
            e_embs = self.energy_embed(e_outs.transpose((0, 2, 1))).transpose(
                (0, 2, 1))
            hs = hs + e_embs + p_embs

            # (B, Lmax, adim)
            hs = self.length_regulator(hs, d_outs, alpha, is_inference=True)
        else:
            d_outs = self.duration_predictor(hs, d_masks)
            # use groundtruth in training
            p_embs = self.pitch_embed(ps.transpose((0, 2, 1))).transpose(
                (0, 2, 1))
            e_embs = self.energy_embed(es.transpose((0, 2, 1))).transpose(
                (0, 2, 1))
            hs = hs + e_embs + p_embs

            # (B, Lmax, adim)
            hs = self.length_regulator(hs, ds, is_inference=False)

        # forward decoder
        if olens is not None and not is_inference:
            if self.reduction_factor > 1:
                olens_in = paddle.to_tensor(
                    [olen // self.reduction_factor for olen in olens.numpy()])
            else:
                olens_in = olens
            # (B, 1, T)
            h_masks = self._source_mask(olens_in)
        else:
            h_masks = None
        if return_after_enc:
            return hs, h_masks

        if self.decoder_type == 'cnndecoder':
            # remove output masks for dygraph to static graph
            zs = self.decoder(hs, h_masks)
            before_outs = zs
        else:
            # (B, Lmax, adim)
            zs, _ = self.decoder(hs, h_masks)
            # (B, Lmax, odim)
            before_outs = self.feat_out(zs).reshape(
                (paddle.shape(zs)[0], -1, self.odim))

        # postnet -> (B, Lmax//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose((0, 2, 1))).transpose((0, 2, 1))

        return before_outs, after_outs, d_outs, p_outs, e_outs, spk_logits

    def encoder_infer(
            self,
            text: paddle.Tensor,
            spk_id=None,
            alpha: float=1.0,
            spk_emb=None,
            tone_id=None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        # input of embedding must be int64
        x = paddle.cast(text, 'int64')
        # setup batch axis
        ilens = paddle.shape(x)[0]

        xs = x.unsqueeze(0)

        if spk_emb is not None:
            spk_emb = spk_emb.unsqueeze(0)

        if tone_id is not None:
            tone_id = tone_id.unsqueeze(0)

        # (1, L, odim)
        # use *_ to avoid bug in dygraph to static graph    
        hs, *_ = self._forward(
            xs,
            ilens,
            is_inference=True,
            return_after_enc=True,
            alpha=alpha,
            spk_emb=spk_emb,
            spk_id=spk_id,
            tone_id=tone_id)
        return hs

    def inference(
            self,
            text: paddle.Tensor,
            durations: paddle.Tensor=None,
            pitch: paddle.Tensor=None,
            energy: paddle.Tensor=None,
            alpha: float=1.0,
            use_teacher_forcing: bool=False,
            spk_emb=None,
            spk_id=None,
            tone_id=None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Generate the sequence of features given the sequences of characters.

        Args:
            text(Tensor(int64)): 
                Input sequence of characters (T,).
            durations(Tensor, optional (int64)): 
                Groundtruth of duration (T,).
            pitch(Tensor, optional): 
                Groundtruth of token-averaged pitch (T, 1).
            energy(Tensor, optional): 
                Groundtruth of token-averaged energy (T, 1).
            alpha(float, optional): 
                Alpha to control the speed.
            use_teacher_forcing(bool, optional): 
                Whether to use teacher forcing.
                If true, groundtruth of duration, pitch and energy will be used.
            spk_emb(Tensor, optional, optional): 
                peaker embedding vector (spk_embed_dim,). (Default value = None)
            spk_id(Tensor, optional(int64), optional): 
                spk ids (1,). (Default value = None)
            tone_id(Tensor, optional(int64), optional): 
                tone ids (T,). (Default value = None)

        Returns:

        """
        # input of embedding must be int64
        x = paddle.cast(text, 'int64')
        d, p, e = durations, pitch, energy
        # setup batch axis
        ilens = paddle.shape(x)[0:1]

        xs = x.unsqueeze(0)

        if spk_emb is not None:
            spk_emb = spk_emb.unsqueeze(0)

        if tone_id is not None:
            tone_id = tone_id.unsqueeze(0)

        if use_teacher_forcing:
            # use groundtruth of duration, pitch, and energy
            ds = d.unsqueeze(0) if d is not None else None
            ps = p.unsqueeze(0) if p is not None else None
            es = e.unsqueeze(0) if e is not None else None

            # (1, L, odim)
            _, outs, d_outs, p_outs, e_outs, _ = self._forward(
                xs,
                ilens,
                ds=ds,
                ps=ps,
                es=es,
                spk_emb=spk_emb,
                spk_id=spk_id,
                tone_id=tone_id,
                is_inference=True)
        else:
            # (1, L, odim)
            _, outs, d_outs, p_outs, e_outs, _ = self._forward(
                xs,
                ilens,
                is_inference=True,
                alpha=alpha,
                spk_emb=spk_emb,
                spk_id=spk_id,
                tone_id=tone_id)

        return outs[0], d_outs[0], p_outs[0], e_outs[0]

    def _integrate_with_spk_embed(self, hs, spk_emb):
        """Integrate speaker embedding with hidden states.

        Args:
            hs(Tensor): 
                Batch of hidden state sequences (B, Tmax, adim).
            spk_emb(Tensor): 
                Batch of speaker embeddings (B, spk_embed_dim).

        Returns:

        
        """
        if self.spk_embed_integration_type == "add":
            # apply projection and then add to hidden states
            spk_emb = self.spk_projection(F.normalize(spk_emb))
            hs = hs + spk_emb.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # one wave `spk_emb` under synthesize, the dim is `1`
            if spk_emb.dim() == 1:
                spk_emb = spk_emb.unsqueeze(0)
            # concat hidden states with spk embeds and then apply projection
            spk_emb = F.normalize(spk_emb).unsqueeze(1).expand(
                shape=[-1, paddle.shape(hs)[1], -1])
            hs = self.spk_projection(paddle.concat([hs, spk_emb], axis=-1))
        else:
            raise NotImplementedError("support only add or concat.")

        return hs

    def _integrate_with_tone_embed(self, hs, tone_embs):
        """Integrate speaker embedding with hidden states.

        Args:
            hs(Tensor): 
                Batch of hidden state sequences (B, Tmax, adim).
            tone_embs(Tensor): 
                Batch of speaker embeddings (B, Tmax, tone_embed_dim).

        Returns:

        """
        if self.tone_embed_integration_type == "add":
            # apply projection and then add to hidden states
            tone_embs = self.tone_projection(F.normalize(tone_embs))
            hs = hs + tone_embs

        elif self.tone_embed_integration_type == "concat":
            # concat hidden states with tone embeds and then apply projection
            tone_embs = F.normalize(tone_embs).expand(
                shape=[-1, hs.shape[1], -1])
            hs = self.tone_projection(paddle.concat([hs, tone_embs], axis=-1))
        else:
            raise NotImplementedError("support only add or concat.")
        return hs

    def _source_mask(self, ilens: paddle.Tensor) -> paddle.Tensor:
        """Make masks for self-attention.

        Args:
            ilens(Tensor): 
                Batch of lengths (B,).

        Returns:
            Tensor: 
                Mask tensor for self-attention. dtype=paddle.bool

        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                        [1, 1, 1, 0, 0]]]) bool
        """
        x_masks = make_non_pad_mask(ilens)
        return x_masks.unsqueeze(-2)

    def _reset_parameters(self, init_enc_alpha: float, init_dec_alpha: float):

        # initialize alpha in scaled positional encoding
        if self.encoder_type == "transformer" and self.use_scaled_pos_enc:
            init_enc_alpha = paddle.to_tensor(init_enc_alpha).reshape([1])
            self.encoder.embed[-1].alpha = paddle.create_parameter(
                shape=init_enc_alpha.shape,
                dtype=str(init_enc_alpha.numpy().dtype),
                default_initializer=paddle.nn.initializer.Assign(
                    init_enc_alpha))
        if self.decoder_type == "transformer" and self.use_scaled_pos_enc:
            init_dec_alpha = paddle.to_tensor(init_dec_alpha).reshape([1])
            self.decoder.embed[-1].alpha = paddle.create_parameter(
                shape=init_dec_alpha.shape,
                dtype=str(init_dec_alpha.numpy().dtype),
                default_initializer=paddle.nn.initializer.Assign(
                    init_dec_alpha))


class FastSpeech2Inference(nn.Layer):
    def __init__(self, normalizer, model):
        super().__init__()
        self.normalizer = normalizer
        self.acoustic_model = model

    def forward(self, text, spk_id=None, spk_emb=None):
        normalized_mel, d_outs, p_outs, e_outs = self.acoustic_model.inference(
            text, spk_id=spk_id, spk_emb=spk_emb)
        logmel = self.normalizer.inverse(normalized_mel)
        return logmel


class StyleFastSpeech2Inference(FastSpeech2Inference):
    def __init__(self,
                 normalizer,
                 model,
                 pitch_stats_path=None,
                 energy_stats_path=None):
        super().__init__(normalizer, model)
        if pitch_stats_path:
            pitch_mean, pitch_std = np.load(pitch_stats_path)
            self.pitch_mean = paddle.to_tensor(pitch_mean)
            self.pitch_std = paddle.to_tensor(pitch_std)
        if energy_stats_path:
            energy_mean, energy_std = np.load(energy_stats_path)
            self.energy_mean = paddle.to_tensor(energy_mean)
            self.energy_std = paddle.to_tensor(energy_std)

    def denorm(self, data, mean, std):
        return data * std + mean

    def norm(self, data, mean, std):
        return (data - mean) / std

    def forward(self,
                text: paddle.Tensor,
                durations: Union[paddle.Tensor, np.ndarray]=None,
                durations_scale: Union[int, float]=None,
                durations_bias: Union[int, float]=None,
                pitch: Union[paddle.Tensor, np.ndarray]=None,
                pitch_scale: Union[int, float]=None,
                pitch_bias: Union[int, float]=None,
                energy: Union[paddle.Tensor, np.ndarray]=None,
                energy_scale: Union[int, float]=None,
                energy_bias: Union[int, float]=None,
                robot: bool=False,
                spk_emb=None,
                spk_id=None):
        """

        Args:
            text(Tensor(int64)): 
                Input sequence of characters (T,).
            durations(paddle.Tensor/np.ndarray, optional (int64)): 
                Groundtruth of duration (T,), this will overwrite the set of durations_scale and durations_bias
            durations_scale(int/float, optional): 

            durations_bias(int/float, optional): 

            pitch(paddle.Tensor/np.ndarray, optional): 
                Groundtruth of token-averaged pitch (T, 1), this will overwrite the set of pitch_scale and pitch_bias
            pitch_scale(int/float, optional): 
                In denormed HZ domain.
            pitch_bias(int/float, optional): 
                In denormed HZ domain.
            energy(paddle.Tensor/np.ndarray, optional): 
                Groundtruth of token-averaged energy (T, 1), this will overwrite the set of energy_scale and energy_bias
            energy_scale(int/float, optional): 
                In denormed domain.
            energy_bias(int/float, optional): 
                In denormed domain.
            robot(bool) (Default value = False):

            spk_emb(Default value = None):

            spk_id(Default value = None):


        Returns:
            Tensor: logmel

        """
        normalized_mel, d_outs, p_outs, e_outs = self.acoustic_model.inference(
            text,
            durations=None,
            pitch=None,
            energy=None,
            spk_emb=spk_emb,
            spk_id=spk_id)
        # priority: groundtruth > scale/bias > previous output
        # set durations
        if isinstance(durations, np.ndarray):
            durations = paddle.to_tensor(durations)
        elif isinstance(durations, paddle.Tensor):
            durations = durations
        elif durations_scale or durations_bias:
            durations_scale = durations_scale if durations_scale is not None else 1
            durations_bias = durations_bias if durations_bias is not None else 0
            durations = durations_scale * d_outs + durations_bias
        else:
            durations = d_outs

        if robot:
            # set normed pitch to zeros have the same effect with set denormd ones to mean
            pitch = paddle.zeros(p_outs.shape)

        # set pitch, can overwrite robot set  
        if isinstance(pitch, np.ndarray):
            pitch = paddle.to_tensor(pitch)
        elif isinstance(pitch, paddle.Tensor):
            pitch = pitch
        elif pitch_scale or pitch_bias:
            pitch_scale = pitch_scale if pitch_scale is not None else 1
            pitch_bias = pitch_bias if pitch_bias is not None else 0
            p_Hz = paddle.exp(
                self.denorm(p_outs, self.pitch_mean, self.pitch_std))
            p_HZ = pitch_scale * p_Hz + pitch_bias
            pitch = self.norm(paddle.log(p_HZ), self.pitch_mean, self.pitch_std)
        else:
            pitch = p_outs

        # set energy
        if isinstance(energy, np.ndarray):
            energy = paddle.to_tensor(energy)
        elif isinstance(energy, paddle.Tensor):
            energy = energy
        elif energy_scale or energy_bias:
            energy_scale = energy_scale if energy_scale is not None else 1
            energy_bias = energy_bias if energy_bias is not None else 0
            e_dnorm = self.denorm(e_outs, self.energy_mean, self.energy_std)
            e_dnorm = energy_scale * e_dnorm + energy_bias
            energy = self.norm(e_dnorm, self.energy_mean, self.energy_std)
        else:
            energy = e_outs

        normalized_mel, d_outs, p_outs, e_outs = self.acoustic_model.inference(
            text,
            durations=durations,
            pitch=pitch,
            energy=energy,
            use_teacher_forcing=True,
            spk_emb=spk_emb,
            spk_id=spk_id)

        logmel = self.normalizer.inverse(normalized_mel)
        return logmel


class FastSpeech2Loss(nn.Layer):
    """Loss function module for FastSpeech2."""

    def __init__(self, use_masking: bool=True,
                 use_weighted_masking: bool=False):
        """Initialize feed-forward Transformer loss module.
        Args:
            use_masking (bool): 
                Whether to apply masking for padded part in loss calculation.
            use_weighted_masking (bool): 
                Whether to weighted masking in loss calculation.
        """
        assert check_argument_types()
        super().__init__()

        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = nn.L1Loss(reduction=reduction)
        self.mse_criterion = nn.MSELoss(reduction=reduction)
        self.duration_criterion = DurationPredictorLoss(reduction=reduction)
        self.ce_criterion = nn.CrossEntropyLoss()

    def forward(
            self,
            after_outs: paddle.Tensor,
            before_outs: paddle.Tensor,
            d_outs: paddle.Tensor,
            p_outs: paddle.Tensor,
            e_outs: paddle.Tensor,
            ys: paddle.Tensor,
            ds: paddle.Tensor,
            ps: paddle.Tensor,
            es: paddle.Tensor,
            ilens: paddle.Tensor,
            olens: paddle.Tensor,
            spk_logits: paddle.Tensor=None,
            spk_ids: paddle.Tensor=None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor,
               paddle.Tensor, ]:
        """Calculate forward propagation.

        Args:
            after_outs(Tensor):  
                Batch of outputs after postnets (B, Lmax, odim).
            before_outs(Tensor): 
                Batch of outputs before postnets (B, Lmax, odim).
            d_outs(Tensor): 
                Batch of outputs of duration predictor (B, Tmax).
            p_outs(Tensor): 
                Batch of outputs of pitch predictor (B, Tmax, 1).
            e_outs(Tensor): 
                Batch of outputs of energy predictor (B, Tmax, 1).
            ys(Tensor): 
                Batch of target features (B, Lmax, odim).
            ds(Tensor): 
                Batch of durations (B, Tmax).
            ps(Tensor): 
                Batch of target token-averaged pitch (B, Tmax, 1).
            es(Tensor): 
                Batch of target token-averaged energy (B, Tmax, 1).
            ilens(Tensor): 
                Batch of the lengths of each input (B,).
            olens(Tensor): 
                Batch of the lengths of each target (B,).
            spk_logits(Option[Tensor]):
                Batch of outputs after speaker classifier (B, Lmax, num_spk)
            spk_ids(Option[Tensor]):
                Batch of target spk_id (B,)
            

        Returns:

        
        """
        speaker_loss = 0.0

        # apply mask to remove padded part
        if self.use_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1)
            before_outs = before_outs.masked_select(
                out_masks.broadcast_to(before_outs.shape))
            if after_outs is not None:
                after_outs = after_outs.masked_select(
                    out_masks.broadcast_to(after_outs.shape))
            ys = ys.masked_select(out_masks.broadcast_to(ys.shape))
            duration_masks = make_non_pad_mask(ilens)
            d_outs = d_outs.masked_select(
                duration_masks.broadcast_to(d_outs.shape))
            ds = ds.masked_select(duration_masks.broadcast_to(ds.shape))
            pitch_masks = make_non_pad_mask(ilens).unsqueeze(-1)
            p_outs = p_outs.masked_select(
                pitch_masks.broadcast_to(p_outs.shape))
            e_outs = e_outs.masked_select(
                pitch_masks.broadcast_to(e_outs.shape))
            ps = ps.masked_select(pitch_masks.broadcast_to(ps.shape))
            es = es.masked_select(pitch_masks.broadcast_to(es.shape))

            if spk_logits is not None and spk_ids is not None:
                batch_size = spk_ids.shape[0]
                spk_ids = paddle.repeat_interleave(spk_ids, spk_logits.shape[1],
                                                   None)
                spk_logits = paddle.reshape(spk_logits,
                                            [-1, spk_logits.shape[-1]])
                mask_index = spk_logits.abs().sum(axis=1) != 0
                spk_ids = spk_ids[mask_index]
                spk_logits = spk_logits[mask_index]

        # calculate loss
        l1_loss = self.l1_criterion(before_outs, ys)
        if after_outs is not None:
            l1_loss += self.l1_criterion(after_outs, ys)
        duration_loss = self.duration_criterion(d_outs, ds)
        pitch_loss = self.mse_criterion(p_outs, ps)
        energy_loss = self.mse_criterion(e_outs, es)

        if spk_logits is not None and spk_ids is not None:
            speaker_loss = self.ce_criterion(spk_logits, spk_ids) / batch_size

        # make weighted mask and apply it
        if self.use_weighted_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1)
            out_weights = out_masks.cast(dtype=paddle.float32) / out_masks.cast(
                dtype=paddle.float32).sum(
                    axis=1, keepdim=True)
            out_weights /= ys.shape[0] * ys.shape[2]
            duration_masks = make_non_pad_mask(ilens)
            duration_weights = (duration_masks.cast(dtype=paddle.float32) /
                                duration_masks.cast(dtype=paddle.float32).sum(
                                    axis=1, keepdim=True))
            duration_weights /= ds.shape[0]

            # apply weight

            l1_loss = l1_loss.multiply(out_weights)
            l1_loss = l1_loss.masked_select(
                out_masks.broadcast_to(l1_loss.shape)).sum()
            duration_loss = (duration_loss.multiply(duration_weights)
                             .masked_select(duration_masks).sum())
            pitch_masks = duration_masks.unsqueeze(-1)
            pitch_weights = duration_weights.unsqueeze(-1)
            pitch_loss = pitch_loss.multiply(pitch_weights)
            pitch_loss = pitch_loss.masked_select(
                pitch_masks.broadcast_to(pitch_loss.shape)).sum()
            energy_loss = energy_loss.multiply(pitch_weights)
            energy_loss = energy_loss.masked_select(
                pitch_masks.broadcast_to(energy_loss.shape)).sum()

        return l1_loss, duration_loss, pitch_loss, energy_loss, speaker_loss
