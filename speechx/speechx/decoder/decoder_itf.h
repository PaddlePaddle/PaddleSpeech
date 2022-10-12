
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
#include "kaldi/decoder/decodable-itf.h"

namespace ppspeech {

class DecoderInterface {
  public:
    virtual ~DecoderInterface() {}

    virtual void InitDecoder() = 0;

    virtual void Reset() = 0;

    virtual void AdvanceDecode(
        const std::shared_ptr<kaldi::DecodableInterface>& decodable) = 0;


    virtual std::string GetFinalBestPath() = 0;

    virtual std::string GetPartialResult() = 0;

    // void Decode();

    // std::string GetBestPath();
    // std::vector<std::pair<double, std::string>> GetNBestPath();

    // int NumFrameDecoded();
    // int DecodeLikelihoods(const std::vector<std::vector<BaseFloat>>& probs,
    //                       std::vector<std::string>& nbest_words);


  private:
    // void AdvanceDecoding(kaldi::DecodableInterface* decodable);

    // current decoding frame number
    int32 num_frame_decoded_;
};

}  // namespace ppspeech