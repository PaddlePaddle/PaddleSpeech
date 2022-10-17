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

using kaldi::BaseFloat;
using kaldi::SubVector;
using kaldi::Vector;
using kaldi::VectorBase;
using std::unique_ptr;
using std::vector;


Recognizer::Recognizer(const RecognizerResource& resource) {
    // resource_ = resource;
    const FeaturePipelineOptions& feature_opts = resource.feature_pipeline_opts;
    feature_pipeline_.reset(new FeaturePipeline(feature_opts));

    std::shared_ptr<PaddleNnet> nnet(new PaddleNnet(resource.model_opts));

    BaseFloat ac_scale = resource.acoustic_scale;
    decodable_.reset(new Decodable(nnet, feature_pipeline_, ac_scale));

    decoder_.reset(new TLGDecoder(resource.tlg_opts));

    input_finished_ = false;
}

void Recognizer::Accept(const Vector<BaseFloat>& waves) {
    feature_pipeline_->Accept(waves);
}

void Recognizer::Decode() { decoder_->AdvanceDecode(decodable_); }

std::string Recognizer::GetFinalResult() {
    return decoder_->GetFinalBestPath();
}

std::string Recognizer::GetPartialResult() {
    return decoder_->GetPartialResult();
}

void Recognizer::SetFinished() {
    feature_pipeline_->SetFinished();
    input_finished_ = true;
}

bool Recognizer::IsFinished() { return input_finished_; }

void Recognizer::Reset() {
    feature_pipeline_->Reset();
    decodable_->Reset();
    decoder_->Reset();
}

}  // namespace ppspeech