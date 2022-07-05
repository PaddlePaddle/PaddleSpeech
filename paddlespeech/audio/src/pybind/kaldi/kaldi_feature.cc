// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddlespeech/audio/src/pybind/kaldi/kaldi_feature.h"
#include "feat/pitch-functions.h"

namespace paddleaudio {
namespace kaldi {

bool InitFbank(float samp_freq,  // frame opts
               float frame_shift_ms,
               float frame_length_ms,
               float dither,
               float preemph_coeff,
               bool remove_dc_offset,
               std::string window_type,  // e.g. Hamming window
               bool round_to_power_of_two,
               float blackman_coeff,
               bool snip_edges,
               bool allow_downsample,
               bool allow_upsample,
               int max_feature_vectors,
               int num_bins,  // mel opts
               float low_freq,
               float high_freq,
               float vtln_low,
               float vtln_high,
               bool debug_mel,
               bool htk_mode,
               bool use_energy,  // fbank opts
               float energy_floor,
               bool raw_energy,
               bool htk_compat,
               bool use_log_fbank,
               bool use_power) {
    ::kaldi::FbankOptions opts;
    opts.frame_opts.samp_freq = samp_freq;  // frame opts
    opts.frame_opts.frame_shift_ms = frame_shift_ms;
    opts.frame_opts.frame_length_ms = frame_length_ms;
    opts.frame_opts.dither = dither;
    opts.frame_opts.preemph_coeff = preemph_coeff;
    opts.frame_opts.remove_dc_offset = remove_dc_offset;
    opts.frame_opts.window_type = window_type;
    opts.frame_opts.round_to_power_of_two = round_to_power_of_two;
    opts.frame_opts.blackman_coeff = blackman_coeff;
    opts.frame_opts.snip_edges = snip_edges;
    opts.frame_opts.allow_downsample = allow_downsample;
    opts.frame_opts.allow_upsample = allow_upsample;
    opts.frame_opts.max_feature_vectors = max_feature_vectors;

    opts.mel_opts.num_bins = num_bins;  // mel opts
    opts.mel_opts.low_freq = low_freq;
    opts.mel_opts.high_freq = high_freq;
    opts.mel_opts.vtln_low = vtln_low;
    opts.mel_opts.vtln_high = vtln_high;
    opts.mel_opts.debug_mel = debug_mel;
    opts.mel_opts.htk_mode = htk_mode;

    opts.use_energy = use_energy;  // fbank opts
    opts.energy_floor = energy_floor;
    opts.raw_energy = raw_energy;
    opts.htk_compat = htk_compat;
    opts.use_log_fbank = use_log_fbank;
    opts.use_power = use_power;
    paddleaudio::kaldi::KaldiFeatureWrapper::GetInstance()->InitFbank(opts);
    return true;
}

py::array_t<double> ComputeFbankStreaming(const py::array_t<double>& wav) {
    return paddleaudio::kaldi::KaldiFeatureWrapper::GetInstance()->ComputeFbank(
        wav);
}

py::array_t<double> ComputeFbank(
    float samp_freq,  // frame opts
    float frame_shift_ms,
    float frame_length_ms,
    float dither,
    float preemph_coeff,
    bool remove_dc_offset,
    std::string window_type,  // e.g. Hamming window
    bool round_to_power_of_two,
    float blackman_coeff,
    bool snip_edges,
    bool allow_downsample,
    bool allow_upsample,
    int max_feature_vectors,
    int num_bins,  // mel opts
    float low_freq,
    float high_freq,
    float vtln_low,
    float vtln_high,
    bool debug_mel,
    bool htk_mode,
    bool use_energy,  // fbank opts
    float energy_floor,
    bool raw_energy,
    bool htk_compat,
    bool use_log_fbank,
    bool use_power,
    const py::array_t<double>& wav) {
    InitFbank(samp_freq,  // frame opts
              frame_shift_ms,
              frame_length_ms,
              dither,
              preemph_coeff,
              remove_dc_offset,
              window_type,  // e.g. Hamming window
              round_to_power_of_two,
              blackman_coeff,
              snip_edges,
              allow_downsample,
              allow_upsample,
              max_feature_vectors,
              num_bins,  // mel opts
              low_freq,
              high_freq,
              vtln_low,
              vtln_high,
              debug_mel,
              htk_mode,
              use_energy,  // fbank opts
              energy_floor,
              raw_energy,
              htk_compat,
              use_log_fbank,
              use_power);
    py::array_t<double> result = ComputeFbankStreaming(wav);
    paddleaudio::kaldi::KaldiFeatureWrapper::GetInstance()->ResetFbank();
    return result;
}


void ResetFbank() {
    paddleaudio::kaldi::KaldiFeatureWrapper::GetInstance()->ResetFbank();
}

py::array_t<double> ComputeKaldiPitch(
 int samp_freq,
 float frame_shift_ms,
 float frame_length_ms,
 float preemph_coeff,
 int min_f0,
 int max_f0,
 float soft_min_f0,
 float penalty_factor,
 int   lowpass_cutoff,
 int  resample_freq,
 float delta_pitch,
 int  nccf_ballast,
 int lowpass_filter_width,
 int upsample_filter_width,
 int  max_frames_latency,
 int  frames_per_chunk,
 bool  simulate_first_pass_online,
 int  recompute_frame,
 bool  nccf_ballast_online,
 bool  snip_edges,
 const py::array_t<double>& wav) {
::kaldi::PitchExtractionOptions opts;
opts.samp_freq = samp_freq;
opts.frame_shift_ms = frame_shift_ms;
opts.frame_length_ms = frame_length_ms;
opts.preemph_coeff = preemph_coeff;
      opts.min_f0 = min_f0;
      opts.max_f0 = max_f0;
      opts.soft_min_f0 = soft_min_f0;
      opts.penalty_factor = penalty_factor;
      opts.lowpass_cutoff = lowpass_cutoff;
      opts.resample_freq = resample_freq;
      opts.delta_pitch = delta_pitch;
      opts.nccf_ballast = nccf_ballast;
      opts.lowpass_filter_width = lowpass_filter_width;
      opts.upsample_filter_width = upsample_filter_width;
      opts.max_frames_latency = max_frames_latency;
      opts.frames_per_chunk = frames_per_chunk;
      opts.simulate_first_pass_online = simulate_first_pass_online;
      opts.recompute_frame = recompute_frame;
      opts.nccf_ballast_online = nccf_ballast_online;
      opts.snip_edges = snip_edges;

   py::buffer_info info = wav.request();
   kaldi::Vector<::kaldi::BaseFloat> input_wav(info.size);
   double* wav_ptr = (double*)info.ptr;
   for (int idx = 0; idx < info.size; ++idx) {
       input_wav(idx) = *wav_ptr;
       wav_ptr++;
   }
   
   kaldi::Matrix<kaldi::BaseFloat> features; 
   kaldi::ComputeKaldiPitch(opts, input_wav, &features);
   auto result = py::array_t<double>({features.NumRows(), features.NumCols()});
   for (int row_idx = 0; row_idx < features.NumRows(); ++row_idx) {
        for (int col_idx = 0; col_idx < features.NumCols(); ++col_idx) {
        result.mutable_at(row_idx, col_idx) = features(row_idx, col_idx);

        }
   }

   return result;
}


}  // namespace kaldi
}  // namespace paddleaudio
