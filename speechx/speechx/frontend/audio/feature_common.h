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

#include "frontend_itf.h"
#include "kaldi/feat/feature-window.h"

namespace ppspeech {

template <class F>
class StreamingFeatureTpl : public FrontendInterface {
  public:
    typedef typename F::Options Options;
    StreamingFeatureTpl(const Options& opts, 
                        std::unique_ptr<FrontendInterface> base_extractor);
    virtual void Accept(const kaldi::VectorBase<kaldi::BaseFloat>& inputs);
    virtual bool Read(kaldi::Vector<kaldi::BaseFloat>* feats);
    
    // the dim_ is the dim of single frame feature
    virtual size_t Dim() const { return computer_.Dim(); }

    virtual void SetFinished() { base_extractor_->SetFinished(); }

    virtual bool IsFinished() const { return base_extractor_->IsFinished(); }

    virtual void Reset() {
        base_extractor_->Reset();
        remained_wav_.Resize(0);
    }
  private:
    bool Compute(const kaldi::Vector<kaldi::BaseFloat>& waves, 
                 kaldi::Vector<kaldi::BaseFloat>* feats);
    Options opts_;
    std::unique_ptr<FrontendInterface> base_extractor_;
    kaldi::FeatureWindowFunction window_function_;
    kaldi::Vector<kaldi::BaseFloat> remained_wav_;
    F computer_;
};

}  // namespace ppspeech

#include "frontend/audio/feature_common_inl.h"