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


#pragma once

#include "base/common.h"
#include "frontend/feature_extractor_interface.h"
#include "kaldi/feat/feature-window.h"

namespace ppspeech {

struct LinearSpectrogramOptions {
    kaldi::FrameExtractionOptions frame_opts;
    kaldi::BaseFloat streaming_chunk;
    LinearSpectrogramOptions() : streaming_chunk(0.36), frame_opts() {}

    void Register(kaldi::OptionsItf* opts) {
        opts->Register(
            "streaming-chunk", &streaming_chunk, "streaming chunk size");
        frame_opts.Register(opts);
    }
};

class LinearSpectrogram : public FeatureExtractorInterface {
  public:
    explicit LinearSpectrogram(
        const LinearSpectrogramOptions& opts,
        std::unique_ptr<FeatureExtractorInterface> base_extractor);
    virtual void AcceptWaveform(
        const kaldi::VectorBase<kaldi::BaseFloat>& input);
    virtual bool Read(kaldi::Vector<kaldi::BaseFloat>* feat);
    virtual size_t Dim() const { return dim_; }
    virtual void SetFinished() { base_extractor_->SetFinished(); }
    virtual bool IsFinished() const { return base_extractor_->IsFinished(); }

  private:
    void Hanning(std::vector<kaldi::BaseFloat>* data) const;
    bool Compute(const std::vector<kaldi::BaseFloat>& wave,
                 std::vector<std::vector<kaldi::BaseFloat>>& feat);
    void Compute(const kaldi::VectorBase<kaldi::BaseFloat>& input,
                 kaldi::VectorBase<kaldi::BaseFloat>* feature);
    bool NumpyFft(std::vector<kaldi::BaseFloat>* v,
                  std::vector<kaldi::BaseFloat>* real,
                  std::vector<kaldi::BaseFloat>* img) const;

    kaldi::int32 fft_points_;
    size_t dim_;
    std::vector<kaldi::BaseFloat> hanning_window_;
    kaldi::BaseFloat hanning_window_energy_;
    LinearSpectrogramOptions opts_;
    kaldi::Vector<kaldi::BaseFloat> waveform_;  // remove later, todo(SmileGoat)
    std::unique_ptr<FeatureExtractorInterface> base_extractor_;
    int chunk_sample_size_;
    DISALLOW_COPY_AND_ASSIGN(LinearSpectrogram);
};


}  // namespace ppspeech