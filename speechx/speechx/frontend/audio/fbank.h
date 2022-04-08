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

// wrap the fbank feat of kaldi, todo (SmileGoat)

#include "kaldi/feat/feature-mfcc.h"

#incldue "kaldi/matrix/kaldi-vector.h"

namespace ppspeech {

class FbankExtractor : FrontendInterface {
  public:
    explicit FbankExtractor(const FbankOptions& opts,
                            share_ptr<FrontendInterface> pre_extractor);
    virtual void AcceptWaveform(
        const kaldi::Vector<kaldi::BaseFloat>& input) = 0;
    virtual void Read(kaldi::Vector<kaldi::BaseFloat>* feat) = 0;
    virtual size_t Dim() const = 0;

  private:
    bool Compute(const kaldi::Vector<kaldi::BaseFloat>& wave,
                 kaldi::Vector<kaldi::BaseFloat>* feat) const;
};

}  // namespace ppspeech