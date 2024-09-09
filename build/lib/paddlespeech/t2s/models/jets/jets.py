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
"""JETS module"""
import math
from typing import Any
from typing import Dict
from typing import Optional

import paddle
from paddle import nn
from typeguard import check_argument_types

from paddlespeech.t2s.models.hifigan import HiFiGANMultiPeriodDiscriminator
from paddlespeech.t2s.models.hifigan import HiFiGANMultiScaleDiscriminator
from paddlespeech.t2s.models.hifigan import HiFiGANMultiScaleMultiPeriodDiscriminator
from paddlespeech.t2s.models.hifigan import HiFiGANPeriodDiscriminator
from paddlespeech.t2s.models.hifigan import HiFiGANScaleDiscriminator
from paddlespeech.t2s.models.jets.generator import JETSGenerator
from paddlespeech.utils.initialize import _calculate_fan_in_and_fan_out
from paddlespeech.utils.initialize import kaiming_uniform_
from paddlespeech.utils.initialize import normal_
from paddlespeech.utils.initialize import ones_
from paddlespeech.utils.initialize import uniform_
from paddlespeech.utils.initialize import zeros_

AVAILABLE_GENERATERS = {
    "jets_generator": JETSGenerator,
}
AVAILABLE_DISCRIMINATORS = {
    "hifigan_period_discriminator":
    HiFiGANPeriodDiscriminator,
    "hifigan_scale_discriminator":
    HiFiGANScaleDiscriminator,
    "hifigan_multi_period_discriminator":
    HiFiGANMultiPeriodDiscriminator,
    "hifigan_multi_scale_discriminator":
    HiFiGANMultiScaleDiscriminator,
    "hifigan_multi_scale_multi_period_discriminator":
    HiFiGANMultiScaleMultiPeriodDiscriminator,
}


class JETS(nn.Layer):
    """JETS module (generator + discriminator).
    This is a module of JETS described in `JETS: Jointly Training FastSpeech2 
    and HiFi-GAN for End to End Text to Speech`_.
    .. _`JETS: Jointly Training FastSpeech2 and HiFi-GAN for End to End Text to Speech
        Text-to-Speech`: https://arxiv.org/abs/2203.16852v1
    """

    def __init__(
            self,
            # generator related
            idim: int,
            odim: int,
            sampling_rate: int=22050,
            generator_type: str="jets_generator",
            generator_params: Dict[str, Any]={
                "adim": 256,
                "aheads": 2,
                "elayers": 4,
                "eunits": 1024,
                "dlayers": 4,
                "dunits": 1024,
                "positionwise_layer_type": "conv1d",
                "positionwise_conv_kernel_size": 1,
                "use_scaled_pos_enc": True,
                "use_batch_norm": True,
                "encoder_normalize_before": True,
                "decoder_normalize_before": True,
                "encoder_concat_after": False,
                "decoder_concat_after": False,
                "reduction_factor": 1,
                "encoder_type": "transformer",
                "decoder_type": "transformer",
                "transformer_enc_dropout_rate": 0.1,
                "transformer_enc_positional_dropout_rate": 0.1,
                "transformer_enc_attn_dropout_rate": 0.1,
                "transformer_dec_dropout_rate": 0.1,
                "transformer_dec_positional_dropout_rate": 0.1,
                "transformer_dec_attn_dropout_rate": 0.1,
                "conformer_rel_pos_type": "latest",
                "conformer_pos_enc_layer_type": "rel_pos",
                "conformer_self_attn_layer_type": "rel_selfattn",
                "conformer_activation_type": "swish",
                "use_macaron_style_in_conformer": True,
                "use_cnn_in_conformer": True,
                "zero_triu": False,
                "conformer_enc_kernel_size": 7,
                "conformer_dec_kernel_size": 31,
                "duration_predictor_layers": 2,
                "duration_predictor_chans": 384,
                "duration_predictor_kernel_size": 3,
                "duration_predictor_dropout_rate": 0.1,
                "energy_predictor_layers": 2,
                "energy_predictor_chans": 384,
                "energy_predictor_kernel_size": 3,
                "energy_predictor_dropout": 0.5,
                "energy_embed_kernel_size": 1,
                "energy_embed_dropout": 0.5,
                "stop_gradient_from_energy_predictor": False,
                "pitch_predictor_layers": 5,
                "pitch_predictor_chans": 384,
                "pitch_predictor_kernel_size": 5,
                "pitch_predictor_dropout": 0.5,
                "pitch_embed_kernel_size": 1,
                "pitch_embed_dropout": 0.5,
                "stop_gradient_from_pitch_predictor": True,
                "generator_out_channels": 1,
                "generator_channels": 512,
                "generator_global_channels": -1,
                "generator_kernel_size": 7,
                "generator_upsample_scales": [8, 8, 2, 2],
                "generator_upsample_kernel_sizes": [16, 16, 4, 4],
                "generator_resblock_kernel_sizes": [3, 7, 11],
                "generator_resblock_dilations":
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "generator_use_additional_convs": True,
                "generator_bias": True,
                "generator_nonlinear_activation": "LeakyReLU",
                "generator_nonlinear_activation_params": {
                    "negative_slope": 0.1
                },
                "generator_use_weight_norm": True,
                "segment_size": 64,
                "spks": -1,
                "langs": -1,
                "spk_embed_dim": None,
                "spk_embed_integration_type": "add",
                "use_gst": False,
                "gst_tokens": 10,
                "gst_heads": 4,
                "gst_conv_layers": 6,
                "gst_conv_chans_list": [32, 32, 64, 64, 128, 128],
                "gst_conv_kernel_size": 3,
                "gst_conv_stride": 2,
                "gst_gru_layers": 1,
                "gst_gru_units": 128,
                "init_type": "xavier_uniform",
                "init_enc_alpha": 1.0,
                "init_dec_alpha": 1.0,
                "use_masking": False,
                "use_weighted_masking": False,
            },
            # discriminator related
            discriminator_type: str="hifigan_multi_scale_multi_period_discriminator",
            discriminator_params: Dict[str, Any]={
                "scales": 1,
                "scale_downsample_pooling": "AvgPool1D",
                "scale_downsample_pooling_params": {
                    "kernel_size": 4,
                    "stride": 2,
                    "padding": 2,
                },
                "scale_discriminator_params": {
                    "in_channels": 1,
                    "out_channels": 1,
                    "kernel_sizes": [15, 41, 5, 3],
                    "channels": 128,
                    "max_downsample_channels": 1024,
                    "max_groups": 16,
                    "bias": True,
                    "downsample_scales": [2, 2, 4, 4, 1],
                    "nonlinear_activation": "leakyrelu",
                    "nonlinear_activation_params": {
                        "negative_slope": 0.1
                    },
                    "use_weight_norm": True,
                    "use_spectral_norm": False,
                },
                "follow_official_norm": False,
                "periods": [2, 3, 5, 7, 11],
                "period_discriminator_params": {
                    "in_channels": 1,
                    "out_channels": 1,
                    "kernel_sizes": [5, 3],
                    "channels": 32,
                    "downsample_scales": [3, 3, 3, 3, 1],
                    "max_downsample_channels": 1024,
                    "bias": True,
                    "nonlinear_activation": "leakyrelu",
                    "nonlinear_activation_params": {
                        "negative_slope": 0.1
                    },
                    "use_weight_norm": True,
                    "use_spectral_norm": False,
                },
            },
            cache_generator_outputs: bool=True, ):
        """Initialize JETS module.
        Args:
            idim (int):
                Input vocabrary size.
            odim (int):
                Acoustic feature dimension. The actual output channels will
                be 1 since JETS is the end-to-end text-to-wave model but for the
                compatibility odim is used to indicate the acoustic feature dimension.
            sampling_rate (int):
                Sampling rate, not used for the training but it will
                be referred in saving waveform during the inference.
            generator_type (str):
                Generator type.
            generator_params (Dict[str, Any]):
                Parameter dict for generator.
            discriminator_type (str):
                Discriminator type.
            discriminator_params (Dict[str, Any]):
                Parameter dict for discriminator.
            cache_generator_outputs (bool):
                Whether to cache generator outputs.
        """
        assert check_argument_types()
        super().__init__()

        # define modules
        generator_class = AVAILABLE_GENERATERS[generator_type]
        if generator_type == "jets_generator":
            # NOTE: Update parameters for the compatibility.
            #   The idim and odim is automatically decided from input data,
            #   where idim represents #vocabularies and odim represents
            #   the input acoustic feature dimension.
            generator_params.update(idim=idim, odim=odim)
        self.generator = generator_class(
            **generator_params, )
        discriminator_class = AVAILABLE_DISCRIMINATORS[discriminator_type]
        self.discriminator = discriminator_class(
            **discriminator_params, )

        # cache
        self.cache_generator_outputs = cache_generator_outputs
        self._cache = None

        # store sampling rate for saving wav file
        # (not used for the training)
        self.fs = sampling_rate

        # store parameters for test compatibility
        self.spks = self.generator.spks
        self.langs = self.generator.langs
        self.spk_embed_dim = self.generator.spk_embed_dim

        self.reuse_cache_gen = True
        self.reuse_cache_dis = True

        self.reset_parameters()
        self.generator._reset_parameters(
            init_type=generator_params["init_type"],
            init_enc_alpha=generator_params["init_enc_alpha"],
            init_dec_alpha=generator_params["init_dec_alpha"], )

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
            forward_generator: bool=True,
            use_alignment_module: bool=False,
            **kwargs,
    ) -> Dict[str, Any]:
        """Perform generator forward.
        Args:
            text (Tensor):
                Text index tensor (B, T_text).
            text_lengths (Tensor):
                Text length tensor (B,).
            feats (Tensor):
                Feature tensor (B, T_feats, aux_channels).
            feats_lengths (Tensor):
                Feature length tensor (B,).
            durations(Tensor(int64)): 
                Batch of padded durations (B, Tmax).
            durations_lengths (Tensor):
                durations length tensor (B,).
            pitch(Tensor): 
                Batch of padded token-averaged pitch (B, Tmax, 1).
            energy(Tensor): 
                Batch of padded token-averaged energy (B, Tmax, 1).
            sids (Optional[Tensor]):
                Speaker index tensor (B,) or (B, 1).
            spembs (Optional[Tensor]):
                Speaker embedding tensor (B, spk_embed_dim).
            lids (Optional[Tensor]):
                Language index tensor (B,) or (B, 1).
            forward_generator (bool):
                Whether to forward generator.
            use_alignment_module (bool):
                Whether to use alignment module.
        Returns:

        """
        if forward_generator:
            return self._forward_generator(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                durations=durations,
                durations_lengths=durations_lengths,
                pitch=pitch,
                energy=energy,
                sids=sids,
                spembs=spembs,
                lids=lids,
                use_alignment_module=use_alignment_module, )
        else:
            return self._forward_discrminator(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                durations=durations,
                durations_lengths=durations_lengths,
                pitch=pitch,
                energy=energy,
                sids=sids,
                spembs=spembs,
                lids=lids,
                use_alignment_module=use_alignment_module, )

    def _forward_generator(
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
            **kwargs, ) -> Dict[str, Any]:
        """Perform generator forward.
        Args:
            text (Tensor):
                Text index tensor (B, T_text).
            text_lengths (Tensor):
                Text length tensor (B,).
            feats (Tensor):
                Feature tensor (B, T_feats, aux_channels).
            feats_lengths (Tensor):
                Feature length tensor (B,).
            durations(Tensor(int64)): 
                Batch of padded durations (B, Tmax).
            durations_lengths (Tensor):
                durations length tensor (B,).
            pitch(Tensor): 
                Batch of padded token-averaged pitch (B, Tmax, 1).
            energy(Tensor): 
                Batch of padded token-averaged energy (B, Tmax, 1).
            sids (Optional[Tensor]):
                Speaker index tensor (B,) or (B, 1).
            spembs (Optional[Tensor]):
                Speaker embedding tensor (B, spk_embed_dim).
            lids (Optional[Tensor]):
                Language index tensor (B,) or (B, 1).
            use_alignment_module (bool):
                Whether to use alignment module.
        Returns:

        """
        # setup
        # calculate generator outputs
        self.reuse_cache_gen = True
        if not self.cache_generator_outputs or self._cache is None:
            self.reuse_cache_gen = False
            outs = self.generator(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                durations=durations,
                durations_lengths=durations_lengths,
                pitch=pitch,
                energy=energy,
                sids=sids,
                spembs=spembs,
                lids=lids,
                use_alignment_module=use_alignment_module, )
        else:
            outs = self._cache

        # store cache
        if self.training and self.cache_generator_outputs and not self.reuse_cache_gen:
            self._cache = outs

        return outs

    def _forward_discrminator(
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
            **kwargs, ) -> Dict[str, Any]:
        """Perform discriminator forward.
        Args:
            text (Tensor):
                Text index tensor (B, T_text).
            text_lengths (Tensor):
                Text length tensor (B,).
            feats (Tensor):
                Feature tensor (B, T_feats, aux_channels).
            feats_lengths (Tensor):
                Feature length tensor (B,).
            durations(Tensor(int64)): 
                Batch of padded durations (B, Tmax).
            durations_lengths (Tensor):
                durations length tensor (B,).
            pitch(Tensor): 
                Batch of padded token-averaged pitch (B, Tmax, 1).
            energy(Tensor): 
                Batch of padded token-averaged energy (B, Tmax, 1).
            sids (Optional[Tensor]):
                Speaker index tensor (B,) or (B, 1).
            spembs (Optional[Tensor]):
                Speaker embedding tensor (B, spk_embed_dim).
            lids (Optional[Tensor]):
                Language index tensor (B,) or (B, 1).
            use_alignment_module (bool):
                Whether to use alignment module.
        Returns:

        """
        # setup
        # calculate generator outputs
        self.reuse_cache_dis = True
        if not self.cache_generator_outputs or self._cache is None:
            self.reuse_cache_dis = False
            outs = self.generator(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                durations=durations,
                durations_lengths=durations_lengths,
                pitch=pitch,
                energy=energy,
                sids=sids,
                spembs=spembs,
                lids=lids,
                use_alignment_module=use_alignment_module,
                **kwargs, )
        else:
            outs = self._cache

        # store cache
        if self.cache_generator_outputs and not self.reuse_cache_dis:
            self._cache = outs

        return outs

    def inference(self,
                  text: paddle.Tensor,
                  feats: Optional[paddle.Tensor]=None,
                  pitch: Optional[paddle.Tensor]=None,
                  energy: Optional[paddle.Tensor]=None,
                  use_alignment_module: bool=False,
                  **kwargs) -> Dict[str, paddle.Tensor]:
        """Run inference.
        Args:
            text (Tensor):
                Input text index tensor (T_text,).
            feats (Tensor):
                Feature tensor (T_feats, aux_channels).
            pitch (Tensor):
                Pitch tensor (T_feats, 1).
            energy (Tensor): 
                Energy tensor (T_feats, 1).
            use_alignment_module (bool):
                Whether to use alignment module.
        Returns:
            Dict[str, Tensor]:
                * wav (Tensor):
                    Generated waveform tensor (T_wav,).
                * duration (Tensor):
                    Predicted duration tensor (T_text,).
        """
        # setup
        text = text[None]
        text_lengths = paddle.to_tensor(paddle.shape(text)[1])

        # inference
        if use_alignment_module:
            assert feats is not None
            feats = feats[None]
            feats_lengths = paddle.to_tensor(paddle.shape(feats)[1])
            pitch = pitch[None]
            energy = energy[None]
            wav, dur = self.generator.inference(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                pitch=pitch,
                energy=energy,
                use_alignment_module=use_alignment_module,
                **kwargs)
        else:
            wav, dur = self.generator.inference(
                text=text,
                text_lengths=text_lengths,
                **kwargs, )
        return dict(wav=paddle.reshape(wav, [-1]), duration=dur[0])

    def reset_parameters(self):
        def _reset_parameters(module):
            if isinstance(
                    module,
                (nn.Conv1D, nn.Conv1DTranspose, nn.Conv2D, nn.Conv2DTranspose)):
                kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = _calculate_fan_in_and_fan_out(module.weight)
                    if fan_in != 0:
                        bound = 1 / math.sqrt(fan_in)
                        uniform_(module.bias, -bound, bound)

            if isinstance(
                    module,
                (nn.BatchNorm1D, nn.BatchNorm2D, nn.GroupNorm, nn.LayerNorm)):
                ones_(module.weight)
                zeros_(module.bias)

            if isinstance(module, nn.Linear):
                kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = _calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    uniform_(module.bias, -bound, bound)

            if isinstance(module, nn.Embedding):
                normal_(module.weight)
                if module._padding_idx is not None:
                    with paddle.no_grad():
                        module.weight[module._padding_idx] = 0

        self.apply(_reset_parameters)


class JETSInference(nn.Layer):
    def __init__(self, model):
        super().__init__()
        self.acoustic_model = model

    def forward(self, text, sids=None):
        out = self.acoustic_model.inference(text)
        wav = out['wav']
        return wav
