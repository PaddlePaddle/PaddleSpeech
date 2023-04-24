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


namespace paddleaudio {
namespace kaldi {

template <class F>
StreamingFeatureTpl<F>::StreamingFeatureTpl(const Options& opts)
    : opts_(opts), computer_(opts), window_function_(opts.frame_opts) {
    // window_function_(computer_.GetFrameOptions()) { the opt set to zero
}

template <class F>
bool StreamingFeatureTpl<F>::ComputeFeature(
    const std::vector<float>& wav,
    std::vector<float>* feats) {
    // append remaned waves
    int wav_len = wav.size();
    if (wav_len == 0) return false;
    int left_len = remained_wav_.size();
    std::vector<float> waves(left_len + wav_len);
    std::memcpy(waves.data(),
                remained_wav_.data(),
                left_len * sizeof(float));
    std::memcpy(waves.data() + left_len,
                wav.data(),
                wav_len * sizeof(float));

    // cache remaned waves
    knf::FrameExtractionOptions frame_opts = computer_.GetFrameOptions();
    int num_frames = knf::NumFrames(waves.size(), frame_opts);
    int frame_shift = frame_opts.WindowShift();
    int left_samples = waves.size() - frame_shift * num_frames;
    remained_wav_.resize(left_samples);
    std::memcpy(remained_wav_.data(),
                waves.data() + frame_shift * num_frames,
                left_samples * sizeof(float));

    // compute speech feature
    Compute(waves, feats);
    return true;
}

// Compute feat
template <class F>
bool StreamingFeatureTpl<F>::Compute(const std::vector<float>& waves,
                                     std::vector<float>* feats) {
    const knf::FrameExtractionOptions& frame_opts = computer_.GetFrameOptions();
    int num_samples = waves.size();
    int frame_length = frame_opts.WindowSize();
    int sample_rate = frame_opts.samp_freq;
    if (num_samples < frame_length) {
        return true;
    }

    int num_frames = knf::NumFrames(num_samples, frame_opts);
    feats->resize(num_frames * Dim());

    std::vector<float> window;
    bool need_raw_log_energy = computer_.NeedRawLogEnergy();
    for (int frame = 0; frame < num_frames; frame++) {
        std::fill(window.begin(), window.end(), 0);
        float raw_log_energy = 0.0;
        float vtln_warp = 1.0;
        knf::ExtractWindow(0,
                           waves,
                           frame,
                           frame_opts,
                           window_function_,
                           &window,
                           need_raw_log_energy ? &raw_log_energy : NULL);

        std::vector<float> this_feature(computer_.Dim());
        computer_.Compute(
            raw_log_energy, vtln_warp, &window, this_feature.data());
        std::memcpy(feats->data() + frame * Dim(),
                    this_feature.data(),
                    sizeof(float) * Dim());
    }
    return true;
}

}  // namespace kaldi
}  // namespace paddleaudio
