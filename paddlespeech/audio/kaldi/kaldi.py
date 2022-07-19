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

__all__ = [
    'fbank',
    'pitch',
]


@module_utils.requires_kaldi()
def fbank(
        wav,
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
        use_energy: bool=False,  # fbank opts
        energy_floor: float=0.0,
        raw_energy: bool=True,
        htk_compat: bool=False,
        use_log_fbank: bool=True,
        use_power: bool=True):
    frame_opts = paddlespeech.audio._paddleaudio.FrameExtractionOptions()
    mel_opts = paddlespeech.audio._paddleaudio.MelBanksOptions()
    fbank_opts = paddlespeech.audio._paddleaudio.FbankOptions()
    frame_opts.samp_freq = samp_freq
    frame_opts.frame_shift_ms = frame_shift_ms
    frame_opts.frame_length_ms = frame_length_ms
    frame_opts.dither = dither
    frame_opts.preemph_coeff = preemph_coeff
    frame_opts.remove_dc_offset = remove_dc_offset
    frame_opts.window_type = window_type
    frame_opts.round_to_power_of_two = round_to_power_of_two
    frame_opts.blackman_coeff = blackman_coeff
    frame_opts.snip_edges = snip_edges
    frame_opts.allow_downsample = allow_downsample
    frame_opts.allow_upsample = allow_upsample
    frame_opts.max_feature_vectors = max_feature_vectors

    mel_opts.num_bins = num_bins
    mel_opts.low_freq = low_freq
    mel_opts.high_freq = high_freq
    mel_opts.vtln_low = vtln_low
    mel_opts.vtln_high = vtln_high
    mel_opts.debug_mel = debug_mel
    mel_opts.htk_mode = htk_mode

    fbank_opts.use_energy = use_energy
    fbank_opts.energy_floor = energy_floor
    fbank_opts.raw_energy = raw_energy
    fbank_opts.htk_compat = htk_compat
    fbank_opts.use_log_fbank = use_log_fbank
    fbank_opts.use_power = use_power
    feat = paddlespeech.audio._paddleaudio.CopmputeFbank(frame_opts, mel_opts, fbank_opts, wav)
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
    pitch_opts = paddlespeech.audio._paddleaudio.PitchExtractionOptions()
    pitch_opts.samp_freq = samp_freq
    pitch_opts.frame_shift_ms = frame_shift_ms
    pitch_opts.frame_length_ms = frame_length_ms
    pitch_opts.preemph_coeff = preemph_coeff
    pitch_opts.min_f0 = min_f0
    pitch_opts.max_f0 = max_f0
    pitch_opts.soft_min_f0 = soft_min_f0
    pitch_opts.penalty_factor = penalty_factor
    pitch_opts.lowpass_cutoff = lowpass_cutoff
    pitch_opts.resample_freq = resample_freq
    pitch_opts.delta_pitch = delta_pitch
    pitch_opts.nccf_ballast = nccf_ballast
    pitch_opts.lowpass_filter_width = lowpass_filter_width
    pitch_opts.upsample_filter_width = upsample_filter_width
    pitch_opts.max_frames_latency = max_frames_latency
    pitch_opts.frames_per_chunk = frames_per_chunk
    pitch_opts.simulate_first_pass_online = simulate_first_pass_online
    pitch_opts.recompute_frame = recompute_frame
    pitch_opts.nccf_ballast_online = nccf_ballast_online
    pitch_opts.snip_edges = snip_edges
    pitch = paddlespeech.audio._paddleaudio.ComputeKaldiPitch(pitch_opts, wav)
    return pitch
