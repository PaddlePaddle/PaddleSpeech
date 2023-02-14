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

#include "common/frontend/data_cache.h"
#include "common/frontend/fbank.h"
#include "common/frontend/feature-fbank.h"
#include "common/frontend/frontend_itf.h"
#include "common/frontend/wave-reader.h"
#include "common/utils/audio_process.h"
#include "common/utils/file_utils.h"
#include "fastdeploy/runtime.h"
#include "kaldi/util/kaldi-io.h"
#include "kaldi/util/table-types.h"

namespace ppspeech {
struct ClsNnetConf {
    // wav
    bool wav_normal_;
    std::string wav_normal_type_;
    float wav_norm_mul_factor_;
    // model
    std::string model_file_path_;
    std::string param_file_path_;
    std::string dict_file_path_;
    int num_cpu_thread_;
    // fbank
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
    int Init(const ClsNnetConf& conf);
    int Forward(const char* wav_path,
                int topk,
                char* result,
                int result_max_len);
    void Reset();

  private:
    int ModelForward(float* features,
                     const int num_frames,
                     const int feat_dim,
                     std::vector<float>* model_out);
    int ModelForwardStream(std::vector<float>* feats);
    int GetTopkResult(int k, const std::vector<float>& model_out);

    ClsNnetConf conf_;
    knf::FbankOptions fbank_opts_;
    std::unique_ptr<fastdeploy::Runtime> runtime_;
    std::vector<std::string> dict_;
    std::stringstream ss_;
};

}  // namespace ppspeech