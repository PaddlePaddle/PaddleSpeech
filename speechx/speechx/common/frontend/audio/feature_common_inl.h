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


namespace ppspeech {

template <class F>
StreamingFeatureTpl<F>::StreamingFeatureTpl(
    const Options& opts, std::unique_ptr<FrontendInterface> base_extractor)
    : opts_(opts), computer_(opts), window_function_(opts.frame_opts) {
    base_extractor_ = std::move(base_extractor);
}

template <class F>
void StreamingFeatureTpl<F>::Accept(
    const std::vector<kaldi::BaseFloat>& waves) {
    base_extractor_->Accept(waves);
}

template <class F>
bool StreamingFeatureTpl<F>::Read(std::vector<kaldi::BaseFloat>* feats) {
    std::vector<kaldi::BaseFloat> wav(base_extractor_->Dim());
    bool flag = base_extractor_->Read(&wav);
    if (flag == false || wav.size() == 0) return false;

    // append remaned waves
    int32 wav_len = wav.size();
    int32 left_len = remained_wav_.size();
    std::vector<kaldi::BaseFloat> waves(left_len + wav_len);
    std::memcpy(waves.data(),
                remained_wav_.data(),
                left_len * sizeof(kaldi::BaseFloat));
    std::memcpy(waves.data() + left_len,
                wav.data(),
                wav_len * sizeof(kaldi::BaseFloat));

    // compute speech feature
    Compute(waves, feats);

    // cache remaned waves
    knf::FrameExtractionOptions frame_opts = computer_.GetFrameOptions();
    int32 num_frames = knf::NumFrames(waves.size(), frame_opts);
    int32 frame_shift = frame_opts.WindowShift();
    int32 left_samples = waves.size() - frame_shift * num_frames;
    remained_wav_.resize(left_samples);
    std::memcpy(remained_wav_.data(),
                waves.data() + frame_shift * num_frames,
                left_samples * sizeof(BaseFloat));
    return true;
}

// Compute feat
template <class F>
bool StreamingFeatureTpl<F>::Compute(const std::vector<kaldi::BaseFloat>& waves,
                                     std::vector<kaldi::BaseFloat>* feats) {
    const knf::FrameExtractionOptions& frame_opts = computer_.GetFrameOptions();
    int32 num_samples = waves.size();
    int32 frame_length = frame_opts.WindowSize();
    int32 sample_rate = frame_opts.samp_freq;
    if (num_samples < frame_length) {
        return true;
    }

    int32 num_frames = knf::NumFrames(num_samples, frame_opts);
    feats->resize(num_frames * Dim());

    std::vector<kaldi::BaseFloat> window;
    bool need_raw_log_energy = computer_.NeedRawLogEnergy();
    for (int32 frame = 0; frame < num_frames; frame++) {
        std::fill(window.begin(), window.end(), 0);
        kaldi::BaseFloat raw_log_energy = 0.0;
        kaldi::BaseFloat vtln_warp = 1.0;
        knf::ExtractWindow(0,
                           waves,
                           frame,
                           frame_opts,
                           window_function_,
                           &window,
                           need_raw_log_energy ? &raw_log_energy : NULL);

        std::vector<kaldi::BaseFloat> this_feature(computer_.Dim());
        computer_.Compute(
            raw_log_energy, vtln_warp, &window, this_feature.data());
        std::memcpy(feats->data() + frame * Dim(),
                    this_feature.data(),
                    sizeof(BaseFloat) * Dim());
    }
    return true;
}

}  // namespace ppspeech
