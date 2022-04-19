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

// todo refactor later (SGoat)

#pragma once

#include "frontend/audio/audio_cache.h"
#include "frontend/audio/data_cache.h"
#include "frontend/audio/feature_cache.h"
#include "frontend/audio/frontend_itf.h"
#include "frontend/audio/linear_spectrogram.h"
#include "frontend/audio/normalizer.h"

namespace ppspeech {

struct FeaturePipelineOptions {
    std::string cmvn_file;
    LinearSpectrogramOptions linear_spectrogram_opts;
    FeatureCacheOptions feature_cache_opts;
    FeaturePipelineOptions() : cmvn_file("cmvn.ark"), linear_spectrogram_opts(), feature_cache_opts() {}
};

class FeaturePipeline : public FrontendInterface {
  public:
    explicit FeaturePipeline(const FeaturePipelineOptions& opts);
    virtual void Accept(const kaldi::VectorBase<kaldi::BaseFloat>& waves) {
        base_extractor_->Accept(waves);
    }
    virtual bool Read(kaldi::Vector<kaldi::BaseFloat>* feats) {
        return base_extractor_->Read(feats);
    }
    virtual size_t Dim() const { return base_extractor_->Dim(); }
    virtual void SetFinished() { base_extractor_->SetFinished(); }
    virtual bool IsFinished() const { return base_extractor_->IsFinished(); }
    virtual void Reset() { base_extractor_->Reset(); }

  private:
    std::unique_ptr<FrontendInterface> base_extractor_;
};
}