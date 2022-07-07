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
from paddlespeech.audio._internal import module_utils 
import paddlespeech.audio.ops.paddleaudio.ComputeFbank as ComputeFbank
import paddlespeech.audio.ops.paddleaudio.ComputeKaldiPitch as ComputeKaldiPitch

__all__ = [
    'fbank',
    'pitch',
]

@module_utils.requires_kaldi()
def fbank(wav,
          samp_freq: int=16000,
          frame_shift_ms: float=10.0,
          frame_length_ms: float=25.0,
          dither: float=0.0,
          preemph_coeff: float=0.97,
          remove_dc_offset: bool=True,
          window_type: str='povey',
          round_to_power_of_two: bool=True,
          blackman_coeff: float=0.42,
          snip_edges: bool=True,
          allow_downsample: bool=False,
          allow_upsample: bool=False,
          max_feature_vectors: int=-1,
          num_bins: int=23,
          low_freq: float=20,
          high_freq: float=0,
          vtln_low: float=100,
          vtln_high: float=-500,
          debug_mel: bool=False,
          htk_mode: bool=False,
          use_energy: bool=False, # fbank opts
          energy_floor: float=0.0,
          raw_energy: bool=True,
          htk_compat: bool=False,
          use_log_fbank: bool=True,
          use_power: bool=True):
    feat = ComputeFbank(
        samp_freq, frame_shift_ms, frame_length_ms,
        dither, preemph_coeff, remove_dc_offset,
        window_type, round_to_power_of_two, blackman_coeff,
        snip_edges, allow_downsample, allow_upsample,
        max_feature_vectors, num_bins, low_freq,
        high_freq, vtln_low, vtln_high, debug_mel,
        htk_mode, use_energy, energy_floor,
        raw_energy, htk_compat, use_log_fbank, use_power, wav)
    return feat

@module_utils.requires_kaldi()
def pitch(wav,
          samp_freq: int=16000,
          frame_shift_ms: float=10.0,
          frame_length_ms: float=25.0,
          preemph_coeff: float=0.0,
          min_f0: int=50,
          max_f0: int=400,
          soft_min_f0: float=10.0,
          penalty_factor: float=0.1,
          lowpass_cutoff: int=1000,
          resample_freq: int=4000,
          delta_pitch: float=0.005,
          nccf_ballast: int=7000,
          lowpass_filter_width: int=1,
          upsample_filter_width: int=5,
          max_frames_latency: int=0,
          frames_per_chunk: int=0,
          simulate_first_pass_online: bool=False,
          recompute_frame: int=500,
          nccf_ballast_online: bool=False,
          snip_edges: bool=True):
    pitch = ComputeKaldiPitch(samp_freq, frame_shift_ms,
          frame_length_ms,
          preemph_coeff,
          min_f0,
          max_f0,
          soft_min_f0,
          penalty_factor,
          lowpass_cutoff,
          resample_freq,
          delta_pitch,
          nccf_ballast,
          lowpass_filter_width,
          upsample_filter_width,
          max_frames_latency,
          frames_per_chunk,
          simulate_first_pass_online,
          recompute_frame,
          nccf_ballast_online,
          snip_edges, wav)
    return pitch