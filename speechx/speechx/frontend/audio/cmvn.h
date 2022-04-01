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

class CMVN : public FrontendInterface {
  public:
    explicit CMVN(std::string cmvn_file,
                  std::unique_ptr<FrontendInterface> base_extractor);
    virtual void Accept(const kaldi::VectorBase<kaldi::BaseFloat>& inputs);

    // the length of feats = feature_row * feature_dim,
    // the Matrix is squashed into Vector
    virtual bool Read(kaldi::Vector<kaldi::BaseFloat>* feats);
    // the dim_ is the feautre dim.
    virtual size_t Dim() const { return dim_; }
    virtual void SetFinished() { base_extractor_->SetFinished(); }
    virtual bool IsFinished() const { return base_extractor_->IsFinished(); }
    virtual void Reset() { base_extractor_->Reset(); }

  private:
    void Compute(kaldi::VectorBase<kaldi::BaseFloat>* feats) const;
    void ApplyCMVN(kaldi::MatrixBase<BaseFloat>* feats);
    kaldi::Matrix<double> stats_;
    std::unique_ptr<FrontendInterface> base_extractor_;
    size_t dim_;
    bool var_norm_;
};

}  // namespace ppspeech