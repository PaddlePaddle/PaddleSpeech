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
#include "nnet/u2_nnet.h"

namespace ppspeech {

RecognizerController::RecognizerController(int num_worker, RecognizerResource resource) {
    recognizer_workers.resize(num_worker);
    for (size_t i = 0; i < num_worker; ++i) {
        recognizer_workers[i].reset(new ppspeech::RecognizerControllerImpl(resource)); 
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
        recognizer_workers[i]->WaitFinished();
    }
}

void RecognizerController::InitDecoder(int idx) {
    recognizer_workers[idx]->InitDecoder();
}

std::string RecognizerController::GetFinalResult(int idx) {
    recognizer_workers[idx]->WaitDecoderFinished();
    recognizer_workers[idx]->AttentionRescoring();
    std::string result = recognizer_workers[idx]->GetFinalResult();
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
