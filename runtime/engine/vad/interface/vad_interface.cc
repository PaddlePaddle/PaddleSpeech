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


#include "vad/interface/vad_interface.h"
#include "common/base/config.h"
#include "vad/nnet/vad.h"


PPSHandle_t PPSVadCreateInstance(const char* conf_path) {
    Config conf(conf_path);
    ppspeech::VadNnetConf nnet_conf;
    nnet_conf.sr = conf.Read("sr", 16000);
    nnet_conf.frame_ms = conf.Read("frame_ms", 32);
    nnet_conf.threshold = conf.Read("threshold", 0.45f);
    nnet_conf.min_silence_duration_ms =
        conf.Read("min_silence_duration_ms", 200);
    nnet_conf.speech_pad_left_ms = conf.Read("speech_pad_left_ms", 0);
    nnet_conf.speech_pad_right_ms = conf.Read("speech_pad_right_ms", 0);

    nnet_conf.model_file_path = conf.Read("model_path", std::string(""));
    nnet_conf.param_file_path = conf.Read("param_path", std::string(""));
    nnet_conf.num_cpu_thread = conf.Read("num_cpu_thread", 1);

    ppspeech::Vad* model = new ppspeech::Vad(nnet_conf.model_file_path);

    // custom config, but must be set before init
    model->SetConfig(nnet_conf);
    model->Init();

    return static_cast<PPSHandle_t>(model);
}


int PPSVadDestroyInstance(PPSHandle_t instance) {
    ppspeech::Vad* model = static_cast<ppspeech::Vad*>(instance);
    if (model != nullptr) {
        delete model;
        model = nullptr;
    }
    return 0;
}

int PPSVadChunkSizeSamples(PPSHandle_t instance) {
    ppspeech::Vad* model = static_cast<ppspeech::Vad*>(instance);
    if (model == nullptr) {
        printf("instance is null\n");
        return -1;
    }

    return model->WindowSizeSamples();
}

PPSVadState_t PPSVadFeedForward(PPSHandle_t instance,
                                float* chunk,
                                int num_element) {
    ppspeech::Vad* model = static_cast<ppspeech::Vad*>(instance);
    if (model == nullptr) {
        printf("instance is null\n");
        return PPS_VAD_ILLEGAL;
    }

    std::vector<float> chunk_in(chunk, chunk + num_element);
    if (!model->ForwardChunk(chunk_in)) {
        printf("forward chunk failed\n");
        return PPS_VAD_ILLEGAL;
    }
    ppspeech::Vad::State s = model->Postprocess();
    PPSVadState_t ret = (PPSVadState_t)s;
    return ret;
}

int PPSVadReset(PPSHandle_t instance) {
    ppspeech::Vad* model = static_cast<ppspeech::Vad*>(instance);
    if (model == nullptr) {
        printf("instance is null\n");
        return -1;
    }
    model->Reset();
    return 0;
}