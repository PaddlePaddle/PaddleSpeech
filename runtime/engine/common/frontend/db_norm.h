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
#include "frontend/audio/frontend_itf.h"
#include "kaldi/matrix/kaldi-matrix.h"
#include "kaldi/util/options-itf.h"

namespace ppspeech {

struct DecibelNormalizerOptions {
    float target_db;
    float max_gain_db;
    bool convert_int_float;
    DecibelNormalizerOptions()
        : target_db(-20), max_gain_db(300.0), convert_int_float(false) {}

    void Register(kaldi::OptionsItf* opts) {
        opts->Register(
            "target-db", &target_db, "target db for db normalization");
        opts->Register(
            "max-gain-db", &max_gain_db, "max gain db for db normalization");
        opts->Register("convert-int-float",
                       &convert_int_float,
                       "if convert int samples to float");
    }
};

class DecibelNormalizer : public FrontendInterface {
  public:
    explicit DecibelNormalizer(
        const DecibelNormalizerOptions& opts,
        std::unique_ptr<FrontendInterface> base_extractor);
    virtual void Accept(const kaldi::VectorBase<kaldi::BaseFloat>& waves);
    virtual bool Read(kaldi::Vector<kaldi::BaseFloat>* waves);
    // noramlize audio, the dim is 1.
    virtual size_t Dim() const { return dim_; }
    virtual void SetFinished() { base_extractor_->SetFinished(); }
    virtual bool IsFinished() const { return base_extractor_->IsFinished(); }
    virtual void Reset() { base_extractor_->Reset(); }

  private:
    bool Compute(kaldi::VectorBase<kaldi::BaseFloat>* waves) const;
    DecibelNormalizerOptions opts_;
    size_t dim_;
    std::unique_ptr<FrontendInterface> base_extractor_;
    kaldi::Vector<kaldi::BaseFloat> waveform_;
};


}  // namespace ppspeech