// Copyright 2022 Horizon Robotics. All Rights Reserved.
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

// modified from
// https://github.com/wenet-e2e/wenet/blob/main/runtime/core/decoder/asr_model.h
#pragma once

#include <iomanip>
#include <algorithm>
#include <numeric>
#include "fastdeploy/runtime.h"
#include "cls/nnet/wav.h"
#include "common/frontend/frontend_itf.h"
#include "common/frontend/data_cache.h"
#include "common/frontend/feature-fbank.h"
#include "frontend/fbank.h"

namespace ppspeech {
struct ClsNnetConf {
    //wav
    bool wav_normal_;
    std::string wav_normal_type_;
    float wav_norm_mul_factor_;
    //model
    std::string model_file_path_;
    std::string param_file_path_;
    std::string dict_file_path_;
    int num_cpu_thread_;
    //fbank
    float samp_freq;
    float frame_length_ms;
    float frame_shift_ms;
    int num_bins;
    float low_freq;
    float high_freq;
    float dither;
};

class ClsNnet {
public:
    ClsNnet();
    int init(ClsNnetConf& conf);
    int forward(const char* wav_path, int topk, char* result, int result_max_len);
    void reset();
private:
    int init_dict(std::string& dict_path);
    int model_forward(const float* features, const int num_frames, const int feat_dim, std::vector<float>& model_out);
    int model_forward_stream(std::vector<float>& feats);
    int get_topk(int k, std::vector<float>& model_out);
    int waveform_float_normal(std::vector<float>& waveform);
    int waveform_normal(std::vector<float>& waveform, bool wav_normal, std::string& wav_normal_type, float wav_norm_mul_factor);
    float power_to_db(float in, float ref_value = 1.0, float amin = 1e-10, float top_db = 80.0);

    ClsNnetConf conf_;
    knf::FbankOptions fbank_opts_;
    std::unique_ptr<fastdeploy::Runtime> runtime_;
    std::vector<std::string> dict_;
    std::stringstream ss_;
};

}  // namespace ppspeech