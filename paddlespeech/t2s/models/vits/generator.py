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
"""Generator module in VITS.

This code is based on https://github.com/jaywalnut310/vits.

"""
import math
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn

from paddlespeech.t2s.models.hifigan import HiFiGANGenerator
from paddlespeech.t2s.models.vits.duration_predictor import StochasticDurationPredictor
from paddlespeech.t2s.models.vits.posterior_encoder import PosteriorEncoder
from paddlespeech.t2s.models.vits.residual_coupling import ResidualAffineCouplingBlock
from paddlespeech.t2s.models.vits.text_encoder import TextEncoder
from paddlespeech.t2s.modules.nets_utils import get_random_segments
from paddlespeech.t2s.modules.nets_utils import make_non_pad_mask


class VITSGenerator(nn.Layer):
    """Generator module in VITS.
    This is a module of VITS described in `Conditional Variational Autoencoder
    with Adversarial Learning for End-to-End Text-to-Speech`_.
    As text encoder, we use conformer architecture instead of the relative positional
    Transformer, which contains additional convolution layers.
    .. _`Conditional Variational Autoencoder with Adversarial Learning for End-to-End
        Text-to-Speech`: https://arxiv.org/abs/2006.04558
    """

    def __init__(
            self,
            vocabs: int,
            aux_channels: int=513,
            hidden_channels: int=192,
            spks: Optional[int]=None,
            langs: Optional[int]=None,
            spk_embed_dim: Optional[int]=None,
            global_channels: int=-1,
            segment_size: int=32,
            text_encoder_attention_heads: int=2,
            text_encoder_ffn_expand: int=4,
            text_encoder_blocks: int=6,
            text_encoder_positionwise_layer_type: str="conv1d",
            text_encoder_positionwise_conv_kernel_size: int=1,
            text_encoder_positional_encoding_layer_type: str="rel_pos",
            text_encoder_self_attention_layer_type: str="rel_selfattn",
            text_encoder_activation_type: str="swish",
            text_encoder_normalize_before: bool=True,
            text_encoder_dropout_rate: float=0.1,
            text_encoder_positional_dropout_rate: float=0.0,
            text_encoder_attention_dropout_rate: float=0.0,
            text_encoder_conformer_kernel_size: int=7,
            use_macaron_style_in_text_encoder: bool=True,
            use_conformer_conv_in_text_encoder: bool=True,
            decoder_kernel_size: int=7,
            decoder_channels: int=512,
            decoder_upsample_scales: List[int]=[8, 8, 2, 2],
            decoder_upsample_kernel_sizes: List[int]=[16, 16, 4, 4],
            decoder_resblock_kernel_sizes: List[int]=[3, 7, 11],
            decoder_resblock_dilations: List[List[int]]=[[1, 3, 5], [1, 3, 5],
                                                         [1, 3, 5]],
            use_weight_norm_in_decoder: bool=True,
            posterior_encoder_kernel_size: int=5,
            posterior_encoder_layers: int=16,
            posterior_encoder_stacks: int=1,
            posterior_encoder_base_dilation: int=1,
            posterior_encoder_dropout_rate: float=0.0,
            use_weight_norm_in_posterior_encoder: bool=True,
            flow_flows: int=4,
            flow_kernel_size: int=5,
            flow_base_dilation: int=1,
            flow_layers: int=4,
            flow_dropout_rate: float=0.0,
            use_weight_norm_in_flow: bool=True,
            use_only_mean_in_flow: bool=True,
            stochastic_duration_predictor_kernel_size: int=3,
            stochastic_duration_predictor_dropout_rate: float=0.5,
            stochastic_duration_predictor_flows: int=4,
            stochastic_duration_predictor_dds_conv_layers: int=3, ):
        """Initialize VITS generator module.
        Args:
            vocabs (int):
                Input vocabulary size.
            aux_channels (int):
                Number of acoustic feature channels.
            hidden_channels (int):
                Number of hidden channels.
            spks (Optional[int]):
                Number of speakers. If set to > 1, assume that the
                sids will be provided as the input and use sid embedding layer.
            langs (Optional[int]):
                Number of languages. If set to > 1, assume that the
                lids will be provided as the input and use sid embedding layer.
            spk_embed_dim (Optional[int]):
                Speaker embedding dimension. If set to > 0,
                assume that spembs will be provided as the input.
            global_channels (int):
                Number of global conditioning channels.
            segment_size (int):
                Segment size for decoder.
            text_encoder_attention_heads (int):
                Number of heads in conformer block of text encoder.
            text_encoder_ffn_expand (int): 
                Expansion ratio of FFN in conformer block of text encoder.
            text_encoder_blocks (int):
                Number of conformer blocks in text encoder.
            text_encoder_positionwise_layer_type (str):
                Position-wise layer type in conformer block of text encoder.
            text_encoder_positionwise_conv_kernel_size (int):
                Position-wise convolution kernel size in conformer block of text encoder. 
                Only used when the above layer type is conv1d or conv1d-linear.
            text_encoder_positional_encoding_layer_type (str):
                Positional encoding layer type in conformer block of text encoder.
            text_encoder_self_attention_layer_type (str):
                Self-attention layer type in conformer block of text encoder.
            text_encoder_activation_type (str):
                Activation function type in conformer block of text encoder.
            text_encoder_normalize_before (bool): 
                Whether to apply layer norm before self-attention in conformer block of text encoder.
            text_encoder_dropout_rate (float):
                Dropout rate in conformer block of text encoder.
            text_encoder_positional_dropout_rate (float):
                Dropout rate for positional encoding in conformer block of text encoder.
            text_encoder_attention_dropout_rate (float):
                Dropout rate for attention in conformer block of text encoder.
            text_encoder_conformer_kernel_size (int):
                Conformer conv kernel size. It will be used when only use_conformer_conv_in_text_encoder = True.
            use_macaron_style_in_text_encoder (bool):
                Whether to use macaron style FFN in conformer block of text encoder.
            use_conformer_conv_in_text_encoder (bool):
                Whether to use covolution in conformer block of text encoder.
            decoder_kernel_size (int):
                Decoder kernel size.
            decoder_channels (int):
                Number of decoder initial channels.
            decoder_upsample_scales (List[int]):
                List of upsampling scales in decoder.
            decoder_upsample_kernel_sizes (List[int]):
                List of kernel size for upsampling layers in decoder.
            decoder_resblock_kernel_sizes (List[int]):
                List of kernel size for resblocks in decoder.
            decoder_resblock_dilations (List[List[int]]):
                List of list of dilations for resblocks in decoder.
            use_weight_norm_in_decoder (bool):
                Whether to apply weight normalization in decoder.
            posterior_encoder_kernel_size (int):
                Posterior encoder kernel size.
            posterior_encoder_layers (int):
                Number of layers of posterior encoder.
            posterior_encoder_stacks (int):
                Number of stacks of posterior encoder.
            posterior_encoder_base_dilation (int):
                Base dilation of posterior encoder.
            posterior_encoder_dropout_rate (float):
                Dropout rate for posterior encoder.
            use_weight_norm_in_posterior_encoder (bool): 
                Whether to apply weight normalization in posterior encoder.
            flow_flows (int):
                Number of flows in flow.
            flow_kernel_size (int):
                Kernel size in flow.
            flow_base_dilation (int):
                Base dilation in flow.
            flow_layers (int):
                Number of layers in flow.
            flow_dropout_rate (float):
                Dropout rate in flow
            use_weight_norm_in_flow (bool):
                Whether to apply weight normalization in flow.
            use_only_mean_in_flow (bool):
                Whether to use only mean in flow.
            stochastic_duration_predictor_kernel_size (int): 
                Kernel size in stochastic duration predictor.
            stochastic_duration_predictor_dropout_rate (float):
                Dropout rate in stochastic duration predictor.
            stochastic_duration_predictor_flows (int):
                Number of flows in stochastic duration predictor.
            stochastic_duration_predictor_dds_conv_layers (int):
                Number of DDS conv layers in stochastic duration predictor.
        """
        super().__init__()
        self.segment_size = segment_size
        self.text_encoder = TextEncoder(
            vocabs=vocabs,
            attention_dim=hidden_channels,
            attention_heads=text_encoder_attention_heads,
            linear_units=hidden_channels * text_encoder_ffn_expand,
            blocks=text_encoder_blocks,
            positionwise_layer_type=text_encoder_positionwise_layer_type,
            positionwise_conv_kernel_size=text_encoder_positionwise_conv_kernel_size,
            positional_encoding_layer_type=text_encoder_positional_encoding_layer_type,
            self_attention_layer_type=text_encoder_self_attention_layer_type,
            activation_type=text_encoder_activation_type,
            normalize_before=text_encoder_normalize_before,
            dropout_rate=text_encoder_dropout_rate,
            positional_dropout_rate=text_encoder_positional_dropout_rate,
            attention_dropout_rate=text_encoder_attention_dropout_rate,
            conformer_kernel_size=text_encoder_conformer_kernel_size,
            use_macaron_style=use_macaron_style_in_text_encoder,
            use_conformer_conv=use_conformer_conv_in_text_encoder, )
        self.decoder = HiFiGANGenerator(
            in_channels=hidden_channels,
            out_channels=1,
            channels=decoder_channels,
            global_channels=global_channels,
            kernel_size=decoder_kernel_size,
            upsample_scales=decoder_upsample_scales,
            upsample_kernel_sizes=decoder_upsample_kernel_sizes,
            resblock_kernel_sizes=decoder_resblock_kernel_sizes,
            resblock_dilations=decoder_resblock_dilations,
            use_weight_norm=use_weight_norm_in_decoder, )
        self.posterior_encoder = PosteriorEncoder(
            in_channels=aux_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            kernel_size=posterior_encoder_kernel_size,
            layers=posterior_encoder_layers,
            stacks=posterior_encoder_stacks,
            base_dilation=posterior_encoder_base_dilation,
            global_channels=global_channels,
            dropout_rate=posterior_encoder_dropout_rate,
            use_weight_norm=use_weight_norm_in_posterior_encoder, )
        self.flow = ResidualAffineCouplingBlock(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            flows=flow_flows,
            kernel_size=flow_kernel_size,
            base_dilation=flow_base_dilation,
            layers=flow_layers,
            global_channels=global_channels,
            dropout_rate=flow_dropout_rate,
            use_weight_norm=use_weight_norm_in_flow,
            use_only_mean=use_only_mean_in_flow, )
        # TODO: Add deterministic version as an option
        self.duration_predictor = StochasticDurationPredictor(
            channels=hidden_channels,
            kernel_size=stochastic_duration_predictor_kernel_size,
            dropout_rate=stochastic_duration_predictor_dropout_rate,
            flows=stochastic_duration_predictor_flows,
            dds_conv_layers=stochastic_duration_predictor_dds_conv_layers,
            global_channels=global_channels, )

        self.upsample_factor = int(np.prod(decoder_upsample_scales))
        self.spks = None
        if spks is not None and spks > 1:
            assert global_channels > 0
            self.spks = spks
            self.global_emb = nn.Embedding(spks, global_channels)
        self.spk_embed_dim = None
        if spk_embed_dim is not None and spk_embed_dim > 0:
            assert global_channels > 0
            self.spk_embed_dim = spk_embed_dim
            self.spemb_proj = nn.Linear(spk_embed_dim, global_channels)
        self.langs = None
        if langs is not None and langs > 1:
            assert global_channels > 0
            self.langs = langs
            self.lang_emb = nn.Embedding(langs, global_channels)

        # delayed import
        from paddlespeech.t2s.models.vits.monotonic_align import maximum_path

        self.maximum_path = maximum_path
        self.pad1d = nn.Pad1D(
            padding=[1, 0],
            mode='constant',
            data_format='NLC', )

    def forward(
            self,
            text: paddle.Tensor,
            text_lengths: paddle.Tensor,
            feats: paddle.Tensor,
            feats_lengths: paddle.Tensor,
            sids: Optional[paddle.Tensor]=None,
            spembs: Optional[paddle.Tensor]=None,
            lids: Optional[paddle.Tensor]=None,
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
            sids (Optional[Tensor]):
                Speaker index tensor (B,) or (B, 1).
            spembs (Optional[Tensor]):
                Speaker embedding tensor (B, spk_embed_dim).
            lids (Optional[Tensor]):
                Language index tensor (B,) or (B, 1).
        Returns:
            Tensor:
                Waveform tensor (B, 1, segment_size * upsample_factor).
            Tensor:
                Duration negative log-likelihood (NLL) tensor (B,).
            Tensor:
                Monotonic attention weight tensor (B, 1, T_feats, T_text).
            Tensor:
                Segments start index tensor (B,).
            Tensor:
                Text mask tensor (B, 1, T_text).
            Tensor: 
                Feature mask tensor (B, 1, T_feats).
                tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
                    - Tensor: Posterior encoder hidden representation (B, H, T_feats).
                    - Tensor: Flow hidden representation (B, H, T_feats).
                    - Tensor: Expanded text encoder projected mean (B, H, T_feats).
                    - Tensor: Expanded text encoder projected scale (B, H, T_feats).
                    - Tensor: Posterior encoder projected mean (B, H, T_feats).
                    - Tensor: Posterior encoder projected scale (B, H, T_feats).
        """
        # forward text encoder
        x, m_p, logs_p, x_mask = self.text_encoder(text, text_lengths)

        # calculate global conditioning
        g = None
        if self.spks is not None:
            # speaker one-hot vector embedding: (B, global_channels, 1)
            g = self.global_emb(paddle.reshape(sids, [-1])).unsqueeze(-1)
        if self.spk_embed_dim is not None:
            # pretreined speaker embedding, e.g., X-vector (B, global_channels, 1)
            g_ = self.spemb_proj(F.normalize(spembs)).unsqueeze(-1)
            if g is None:
                g = g_
            else:
                g = g + g_
        if self.langs is not None:
            # language one-hot vector embedding: (B, global_channels, 1)
            g_ = self.lang_emb(paddle.reshape(lids, [-1])).unsqueeze(-1)
            if g is None:
                g = g_
            else:
                g = g + g_

        # forward posterior encoder
        z, m_q, logs_q, y_mask = self.posterior_encoder(
            feats, feats_lengths, g=g)

        # forward flow
        # (B, H, T_feats)
        z_p = self.flow(z, y_mask, g=g)

        # monotonic alignment search
        with paddle.no_grad():
            # negative cross-entropy
            # (B, H, T_text)
            s_p_sq_r = paddle.exp(-2 * logs_p)
            # (B, 1, T_text)
            tmp1 = -0.5 * math.log(2 * math.pi) - logs_p
            neg_x_ent_1 = paddle.sum(
                tmp1,
                [1],
                keepdim=True, )
            # (B, T_feats, H) x (B, H, T_text) = (B, T_feats, T_text)
            neg_x_ent_2 = paddle.matmul(
                -0.5 * (z_p**2).transpose([0, 2, 1]),
                s_p_sq_r, )
            # (B, T_feats, H) x (B, H, T_text) = (B, T_feats, T_text)
            neg_x_ent_3 = paddle.matmul(
                z_p.transpose([0, 2, 1]),
                (m_p * s_p_sq_r), )
            # (B, 1, T_text)
            tmp2 = -0.5 * (m_p**2) * s_p_sq_r
            neg_x_ent_4 = paddle.sum(
                tmp2,
                [1],
                keepdim=True, )
            # (B, T_feats, T_text)
            neg_x_ent = neg_x_ent_1 + neg_x_ent_2 + neg_x_ent_3 + neg_x_ent_4
            # (B, 1, T_feats, T_text)
            attn_mask = paddle.unsqueeze(x_mask, 2) * paddle.unsqueeze(y_mask,
                                                                       -1)
            # monotonic attention weight: (B, 1, T_feats, T_text)
            attn = (self.maximum_path(
                neg_x_ent,
                attn_mask.squeeze(1), ).unsqueeze(1).detach())

        # forward duration predictor
        # (B, 1, T_text)
        w = attn.sum(2)
        dur_nll = self.duration_predictor(x, x_mask, w=w, g=g)
        dur_nll = dur_nll / paddle.sum(x_mask)
        # expand the length to match with the feature sequence
        # (B, T_feats, T_text) x (B, T_text, H) -> (B, H, T_feats)
        m_p = paddle.matmul(attn.squeeze(1),
                            m_p.transpose([0, 2, 1])).transpose([0, 2, 1])
        # (B, T_feats, T_text) x (B, T_text, H) -> (B, H, T_feats)
        logs_p = paddle.matmul(attn.squeeze(1),
                               logs_p.transpose([0, 2, 1])).transpose([0, 2, 1])

        # get random segments
        z_segments, z_start_idxs = get_random_segments(
            z,
            feats_lengths,
            self.segment_size, )

        # forward decoder with random segments
        wav = self.decoder(z_segments, g=g)

        return (wav, dur_nll, attn, z_start_idxs, x_mask, y_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q), )

    def inference(
            self,
            text: paddle.Tensor,
            text_lengths: paddle.Tensor,
            feats: Optional[paddle.Tensor]=None,
            feats_lengths: Optional[paddle.Tensor]=None,
            sids: Optional[paddle.Tensor]=None,
            spembs: Optional[paddle.Tensor]=None,
            lids: Optional[paddle.Tensor]=None,
            dur: Optional[paddle.Tensor]=None,
            noise_scale: float=0.667,
            noise_scale_dur: float=0.8,
            alpha: float=1.0,
            max_len: Optional[int]=None,
            use_teacher_forcing: bool=False,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Run inference.
        Args:
            text (Tensor):
                Input text index tensor (B, T_text,).
            text_lengths (Tensor):
                Text length tensor (B,).
            feats (Tensor):
                Feature tensor (B, aux_channels, T_feats,).
            feats_lengths (Tensor):
                Feature length tensor (B,).
            sids (Optional[Tensor]):
                Speaker index tensor (B,) or (B, 1).
            spembs (Optional[Tensor]):
                Speaker embedding tensor (B, spk_embed_dim).
            lids (Optional[Tensor]):
                Language index tensor (B,) or (B, 1).
            dur (Optional[Tensor]):
                Ground-truth duration (B, T_text,). If provided,
                skip the prediction of durations (i.e., teacher forcing).
            noise_scale (float):
                Noise scale parameter for flow.
            noise_scale_dur (float):
                Noise scale parameter for duration predictor.
            alpha (float):
                Alpha parameter to control the speed of generated speech.
            max_len (Optional[int]):
                Maximum length of acoustic feature sequence.
            use_teacher_forcing (bool):
                Whether to use teacher forcing.
        Returns:
            Tensor: 
                Generated waveform tensor (B, T_wav).
            Tensor:
                Monotonic attention weight tensor (B, T_feats, T_text).
            Tensor:
                Duration tensor (B, T_text).
        """
        # encoder
        x, m_p, logs_p, x_mask = self.text_encoder(text, text_lengths)
        g = None
        if self.spks is not None:
            # (B, global_channels, 1)
            g = self.global_emb(paddle.reshape(sids, [-1])).unsqueeze(-1)
        if self.spk_embed_dim is not None:
            # (B, global_channels, 1)
            g_ = self.spemb_proj(F.normalize(spembs.unsqueeze(0))).unsqueeze(-1)
            if g is None:
                g = g_
            else:
                g = g + g_
        if self.langs is not None:
            # (B, global_channels, 1)
            g_ = self.lang_emb(paddle.reshape(lids, [-1])).unsqueeze(-1)
            if g is None:
                g = g_
            else:
                g = g + g_

        if use_teacher_forcing:
            # forward posterior encoder
            z, m_q, logs_q, y_mask = self.posterior_encoder(
                feats, feats_lengths, g=g)

            # forward flow
            # (B, H, T_feats)
            z_p = self.flow(z, y_mask, g=g)

            # monotonic alignment search
            # (B, H, T_text)
            s_p_sq_r = paddle.exp(-2 * logs_p)
            # (B, 1, T_text)
            tmp3 = -0.5 * math.log(2 * math.pi) - logs_p
            neg_x_ent_1 = paddle.sum(
                tmp3,
                [1],
                keepdim=True, )
            # (B, T_feats, H) x (B, H, T_text) = (B, T_feats, T_text)
            neg_x_ent_2 = paddle.matmul(
                -0.5 * (z_p**2).transpose([0, 2, 1]),
                s_p_sq_r, )
            # (B, T_feats, H) x (B, H, T_text) = (B, T_feats, T_text)
            neg_x_ent_3 = paddle.matmul(
                z_p.transpose([0, 2, 1]),
                (m_p * s_p_sq_r), )
            # (B, 1, T_text)
            tmp4 = -0.5 * (m_p**2) * s_p_sq_r
            neg_x_ent_4 = paddle.sum(
                tmp4,
                [1],
                keepdim=True, )
            # (B, T_feats, T_text)
            neg_x_ent = neg_x_ent_1 + neg_x_ent_2 + neg_x_ent_3 + neg_x_ent_4
            # (B, 1, T_feats, T_text)
            attn_mask = paddle.unsqueeze(x_mask, 2) * paddle.unsqueeze(y_mask,
                                                                       -1)
            # monotonic attention weight: (B, 1, T_feats, T_text)
            attn = self.maximum_path(
                neg_x_ent,
                attn_mask.squeeze(1), ).unsqueeze(1)
            # (B, 1, T_text)
            dur = attn.sum(2)

            # forward decoder with random segments
            wav = self.decoder(z * y_mask, g=g)
        else:
            # duration
            if dur is None:
                logw = self.duration_predictor(
                    x,
                    x_mask,
                    g=g,
                    inverse=True,
                    noise_scale=noise_scale_dur, )
                w = paddle.exp(logw) * x_mask * alpha
                dur = paddle.ceil(w)
            y_lengths = paddle.cast(
                paddle.clip(paddle.sum(dur, [1, 2]), min=1), dtype='int64')
            y_mask = make_non_pad_mask(y_lengths).unsqueeze(1)
            tmp_a = paddle.cast(paddle.unsqueeze(x_mask, 2), dtype='int64')
            tmp_b = paddle.cast(paddle.unsqueeze(y_mask, -1), dtype='int64')
            attn_mask = tmp_a * tmp_b
            attn = self._generate_path(dur, attn_mask)

            # expand the length to match with the feature sequence
            # (B, T_feats, T_text) x (B, T_text, H) -> (B, H, T_feats)
            m_p = paddle.matmul(
                attn.squeeze(1),
                m_p.transpose([0, 2, 1]), ).transpose([0, 2, 1])
            # (B, T_feats, T_text) x (B, T_text, H) -> (B, H, T_feats)
            logs_p = paddle.matmul(
                attn.squeeze(1),
                logs_p.transpose([0, 2, 1]), ).transpose([0, 2, 1])

            # decoder
            z_p = m_p + paddle.randn(
                paddle.shape(m_p)) * paddle.exp(logs_p) * noise_scale
            z = self.flow(z_p, y_mask.astype(z_p.dtype), g=g, inverse=True)
            wav = self.decoder(
                (z * y_mask.astype(z.dtype))[:, :, :max_len], g=g)

        return wav.squeeze(1), attn.squeeze(1), dur.squeeze(1)

    def voice_conversion(
            self,
            feats: paddle.Tensor=None,
            feats_lengths: paddle.Tensor=None,
            sids_src: Optional[paddle.Tensor]=None,
            sids_tgt: Optional[paddle.Tensor]=None,
            spembs_src: Optional[paddle.Tensor]=None,
            spembs_tgt: Optional[paddle.Tensor]=None,
            lids: Optional[paddle.Tensor]=None, ) -> paddle.Tensor:
        """Run voice conversion.
        Args:
            feats (Tensor):
                Feature tensor (B, aux_channels, T_feats,).
            feats_lengths (Tensor):
                Feature length tensor (B,).
            sids_src (Optional[Tensor]):
                Speaker index tensor of source feature (B,) or (B, 1).
            sids_tgt (Optional[Tensor]):
                Speaker index tensor of target feature (B,) or (B, 1).
            spembs_src (Optional[Tensor]):
                Speaker embedding tensor of source feature (B, spk_embed_dim).
            spembs_tgt (Optional[Tensor]):
                Speaker embedding tensor of target feature (B, spk_embed_dim).
            lids (Optional[Tensor]):
                Language index tensor (B,) or (B, 1).
        Returns:
            Tensor:
                Generated waveform tensor (B, T_wav).
        """
        # encoder
        g_src = None
        g_tgt = None
        if self.spks is not None:
            # (B, global_channels, 1)
            g_src = self.global_emb(
                paddle.reshape(sids_src, [-1])).unsqueeze(-1)
            g_tgt = self.global_emb(
                paddle.reshape(sids_tgt, [-1])).unsqueeze(-1)

        if self.spk_embed_dim is not None:
            # (B, global_channels, 1)
            g_src_ = self.spemb_proj(
                F.normalize(spembs_src.unsqueeze(0))).unsqueeze(-1)
            if g_src is None:
                g_src = g_src_
            else:
                g_src = g_src + g_src_

            # (B, global_channels, 1)
            g_tgt_ = self.spemb_proj(
                F.normalize(spembs_tgt.unsqueeze(0))).unsqueeze(-1)
            if g_tgt is None:
                g_tgt = g_tgt_
            else:
                g_tgt = g_tgt + g_tgt_

        if self.langs is not None:
            # (B, global_channels, 1)
            g_ = self.lang_emb(paddle.reshape(lids, [-1])).unsqueeze(-1)

            if g_src is None:
                g_src = g_
            else:
                g_src = g_src + g_

            if g_tgt is None:
                g_tgt = g_
            else:
                g_tgt = g_tgt + g_

        # forward posterior encoder
        z, m_q, logs_q, y_mask = self.posterior_encoder(
            feats, feats_lengths, g=g_src)

        # forward flow
        # (B, H, T_feats)
        z_p = self.flow(z, y_mask, g=g_src)

        # decoder
        z_hat = self.flow(z_p, y_mask, g=g_tgt, inverse=True)
        wav = self.decoder(z_hat * y_mask, g=g_tgt)

        return wav.squeeze(1)

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
        return path.unsqueeze(1).transpose(
            [0, 1, 3, 2]) * mask.astype(path.dtype)
