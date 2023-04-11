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

#pragma once

#include <queue>
#include <memory>

#include "recognizer/recognizer_controller_impl.h"
#include "nnet/u2_nnet.h"

namespace ppspeech {

class RecognizerController {
  public:
    explicit RecognizerController(int num_worker, RecognizerResource resource);  
    ~RecognizerController();
    int GetRecognizerInstanceId();
    void InitDecoder(int idx);
    void Accept(std::vector<float> data, int idx);
    void SetInputFinished(int idx);
    std::string GetFinalResult(int idx);
    
  private:
    std::queue<int> waiting_workers;  
    std::shared_ptr<ppspeech::U2Nnet> nnet_;
    std::mutex mutex_;
    std::vector<std::unique_ptr<ppspeech::RecognizerControllerImpl>> recognizer_workers;
  
    DISALLOW_COPY_AND_ASSIGN(RecognizerController);
};

}