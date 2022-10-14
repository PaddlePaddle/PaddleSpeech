
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

enum SearchType {
    kPrefixBeamSearch = 0,
    kWfstBeamSearch = 1,
};
class DecoderInterface {
  public:
    virtual ~DecoderInterface() {}

    virtual void InitDecoder() = 0;

    virtual void Reset() = 0;

    // call AdvanceDecoding
    virtual void AdvanceDecode(
        const std::shared_ptr<kaldi::DecodableInterface>& decodable) = 0;

    // call GetBestPath
    virtual std::string GetFinalBestPath() = 0;

    virtual std::string GetPartialResult() = 0;

  protected:
    // virtual void AdvanceDecoding(kaldi::DecodableInterface* decodable) = 0;

    // virtual void Decode() = 0;

    virtual std::string GetBestPath() = 0;

    virtual std::vector<std::pair<double, std::string>> GetNBestPath() = 0;

    virtual std::vector<std::pair<double, std::string>> GetNBestPath(int n) = 0;

    // start from one
    int NumFrameDecoded() { return num_frame_decoded_ + 1; }

  protected:
    // current decoding frame number, abs_time_step_
    int32 num_frame_decoded_;
};

}  // namespace ppspeech