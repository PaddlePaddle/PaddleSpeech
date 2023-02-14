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

#include "cls/nnet/panns_interface.h"
#include "cls/nnet/panns_nnet.h"
#include "common/base/config.h"

namespace ppspeech {

void* ClsCreateInstance(const char* conf_path) {
    Config conf(conf_path);
    // cls init
    ppspeech::ClsNnetConf cls_nnet_conf;
    cls_nnet_conf.wav_normal_ = conf.Read("wav_normal", true);
    cls_nnet_conf.wav_normal_type_ =
        conf.Read("wav_normal_type", std::string("linear"));
    cls_nnet_conf.wav_norm_mul_factor_ = conf.Read("wav_norm_mul_factor", 1.0);
    cls_nnet_conf.model_file_path_ = conf.Read("model_path", std::string(""));
    cls_nnet_conf.param_file_path_ = conf.Read("param_path", std::string(""));
    cls_nnet_conf.dict_file_path_ = conf.Read("dict_path", std::string(""));
    cls_nnet_conf.num_cpu_thread_ = conf.Read("num_cpu_thread", 12);
    cls_nnet_conf.samp_freq = conf.Read("samp_freq", 32000);
    cls_nnet_conf.frame_length_ms = conf.Read("frame_length_ms", 32);
    cls_nnet_conf.frame_shift_ms = conf.Read("frame_shift_ms", 10);
    cls_nnet_conf.num_bins = conf.Read("num_bins", 64);
    cls_nnet_conf.low_freq = conf.Read("low_freq", 50);
    cls_nnet_conf.high_freq = conf.Read("high_freq", 14000);
    cls_nnet_conf.dither = conf.Read("dither", 0.0);

    ppspeech::ClsNnet* cls_model = new ppspeech::ClsNnet();
    int ret = cls_model->Init(cls_nnet_conf);
    return static_cast<void*>(cls_model);
}

int ClsDestroyInstance(void* instance) {
    ppspeech::ClsNnet* cls_model = static_cast<ppspeech::ClsNnet*>(instance);
    if (cls_model != NULL) {
        delete cls_model;
        cls_model = NULL;
    }
    return 0;
}

int ClsFeedForward(void* instance,
                   const char* wav_path,
                   int topk,
                   char* result,
                   int result_max_len) {
    ppspeech::ClsNnet* cls_model = static_cast<ppspeech::ClsNnet*>(instance);
    if (cls_model == NULL) {
        printf("instance is null\n");
        return -1;
    }
    int ret = cls_model->Forward(wav_path, topk, result, result_max_len);
    return 0;
}

int ClsReset(void* instance) {
    ppspeech::ClsNnet* cls_model = static_cast<ppspeech::ClsNnet*>(instance);
    if (cls_model == NULL) {
        printf("instance is null\n");
        return -1;
    }
    cls_model->Reset();
    return 0;
}
}  // namespace ppspeech