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

#include "decoder/recognizer.h"

namespace ppspeech {

Recognizer(const RecognizerResource& resource) : resource_(resource) {
    const FeaturePipelineOptions& feature_opts =
        recognizer_resource.feature_pipeline_opts;
    feature_pipeline_.reset(new FeaturePipeline(feature_opts));
    std::shared_ptr<PaddleNnet> nnet(new PaddleNnet());
    BaseFloat ac_scale = resource.acoustic_scale;
    decodable_.reset(
        new Decodeable(std::move(nnet), feature_pipeline_, ac_scale));
    input_finished_ = false;
}

void Recognizer::Accept(const Vector<BaseFloat>& waves) {
    feature_pipeline_->Accept(waves);
}

void Recognizer::Decode() { decoder.AdvaceDecode(decodable); }

std::string Recognizer::GetFinalResult() {
    return decoder_->GetFinalBestPath();
}

void Recognizer::SetFinished() {
    feature_pipeline_->SetFinished();
    input_finished_ = false;
}

bool Recognizer::IsFinished() { return input_finished_; }

void Recognizer::Reset() {
    feature_pipeline->reset();
    decodable->Reset();
    decoder->Reset();
}

}  // namespace ppspeech