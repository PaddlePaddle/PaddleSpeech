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
#include "kaldi/feat/feature-fbank.h"
#include "kaldi/feat/feature-mfcc.h"
#include "kaldi/matrix/kaldi-vector.h"

namespace ppspeech {

struct FbankOptions {
    kaldi::FbankOptions fbank_opts;
    kaldi::BaseFloat streaming_chunk;  // second

    FbankOptions() : streaming_chunk(0.1), fbank_opts() {}

    void Register(kaldi::OptionsItf* opts) {
        opts->Register("streaming-chunk",
                       &streaming_chunk,
                       "streaming chunk size, default: 0.1 sec");
        fbank_opts.Register(opts);
    }
};


class Fbank : public FrontendInterface {
  public:
    explicit Fbank(const FbankOptions& opts,
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

    FbankOptions opts_;
    std::unique_ptr<FrontendInterface> base_extractor_;

    kaldi::FeatureWindowFunction window_function_;
    kaldi::FbankComputer computer_;
    // features_ is the Mfcc or Plp or Fbank features that we have already
    // computed.
    kaldi::Vector<kaldi::BaseFloat> features_;
    kaldi::Vector<kaldi::BaseFloat> remained_wav_;
    kaldi::int32 chunk_sample_size_;

    DISALLOW_COPY_AND_ASSIGN(Fbank);
};

}  // namespace ppspeech