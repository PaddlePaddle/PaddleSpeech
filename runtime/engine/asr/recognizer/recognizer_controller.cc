// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "recognizer/recognizer_controller.h"
#include "recognizer/u2_recognizer.h"
#include "nnet/u2_nnet.h"

namespace ppspeech {

RecognizerController::RecognizerController(int num_worker, U2RecognizerResource resource) {
    nnet_ = std::make_shared<ppspeech::U2Nnet>(resource.model_opts); 
    recognizer_workers.resize(num_worker);
    for (size_t i = 0; i < num_worker; ++i) {
        recognizer_workers[i].reset(new ppspeech::U2Recognizer(resource, nnet_->Clone())); 
        recognizer_workers[i]->InitDecoder();
        waiting_workers.push(i);
    }
}

int RecognizerController::GetRecognizerInstanceId() {
    if (waiting_workers.empty()) {
        return -1;
    }
    int idx = -1;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        idx = waiting_workers.front();
        waiting_workers.pop();
    }
    return idx;
}

RecognizerController::~RecognizerController() {
    for (size_t i = 0; i < recognizer_workers.size(); ++i) {
        recognizer_workers[i]->SetInputFinished();
        recognizer_workers[i]->WaitDecodeFinished();
    }
}

std::string RecognizerController::GetFinalResult(int idx) {
    recognizer_workers[idx]->WaitDecodeFinished();
    recognizer_workers[idx]->AttentionRescoring();
    std::string result = recognizer_workers[idx]->GetFinalResult();
    recognizer_workers[idx]->InitDecoder();
    {
        std::unique_lock<std::mutex> lock(mutex_);
        waiting_workers.push(idx);
    }
    return result;
}

void RecognizerController::Accept(std::vector<float> data, int idx) {
    recognizer_workers[idx]->Accept(data);
}

void RecognizerController::SetInputFinished(int idx) {
    recognizer_workers[idx]->SetInputFinished();
}

}