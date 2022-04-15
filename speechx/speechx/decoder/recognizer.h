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

#include "decoder/ctc_beam_search_decoder.h"
#include "decoder/ctc_tlg_decoder.h"
#include "frontend/audio/feature_pipeline.h"
#include "nnet/decodable.h"
#include "nnet/paddle_nnet.h"

namespace ppspeech {

struct RecognizerResource {
    FeaturePipelineOptions feature_pipeline_opts;
    ModelOptions model_opts;
    TLGDecoderOptions tlg_opts;
    //    CTCBeamSearchOptions beam_search_opts;
    kaldi::BaseFloat acoustic_scale;
};

class Recognizer {
  public:
    explicit Recognizer(const RecognizerResource& resouce);
    void Accept(kaldi::Vector<int16> waves);
    void Decode();
    std::string GetFinalResult();
    void SetFinished();
    bool IsFinished();
    void Reset();

  private:
    RecognizerResource resource_;
    std::shared_ptr<FeaturePipeline> feature_pipeline_;
    std::unique_ptr<Decodable> decodable_;
    std::unique_ptr<TLGDecoder> decoder_;
    bool input_finished_;
};

}  // namespace ppspeech