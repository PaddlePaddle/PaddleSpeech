// Copyright (c) 2023 Chen Qianhe Authors. All Rights Reserved.
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
#include <iostream>
#include <mutex>
#include <vector>
#include "fastdeploy/fastdeploy_model.h"
#include "fastdeploy/runtime.h"
#include "vad/frontend/wav.h"

namespace ppspeech {

struct VadNnetConf {
    // wav
    int sr;
    int frame_ms;
    float threshold;
    int min_silence_duration_ms;
    int speech_pad_left_ms;
    int speech_pad_right_ms;

    // model
    std::string model_file_path;
    std::string param_file_path;
    std::string dict_file_path;
    int num_cpu_thread;   // 1 thred
    std::string backend;  // ort,lite, etc.
};

class Vad : public fastdeploy::FastDeployModel {
  public:
    enum class State { ILLEGAL = 0, SIL, START, SPEECH, END };
    friend std::ostream& operator<<(std::ostream& os, const Vad::State& s);

    Vad(const std::string& model_file,
        const fastdeploy::RuntimeOption& custom_option =
            fastdeploy::RuntimeOption());

    void Init();

    void Reset();

    void SetConfig(const int& sr,
                   const int& frame_ms,
                   const float& threshold,
                   const int& min_silence_duration_ms,
                   const int& speech_pad_left_ms,
                   const int& speech_pad_right_ms);
    void SetConfig(const VadNnetConf conf);

    bool ForwardChunk(std::vector<float>& chunk);

    const State& Postprocess();

    const std::vector<std::map<std::string, float>> GetResult(
        float removeThreshold = 0.0,
        float expandHeadThreshold = 0.0,
        float expandTailThreshold = 0,
        float mergeThreshold = 0.0) const;

    const std::vector<State> GetStates() const { return states_; }

    int SampleRate() const { return sample_rate_; }

    int FrameMs() const { return frame_ms_; }
    int64_t WindowSizeSamples() const { return window_size_samples_; }

    float Threshold() const { return threshold_; }

    int MinSilenceDurationMs() const {
        return min_silence_samples_ / sample_rate_;
    }
    int SpeechPadLeftMs() const {
        return speech_pad_left_samples_ / sample_rate_;
    }
    int SpeechPadRightMs() const {
        return speech_pad_right_samples_ / sample_rate_;
    }

    int MinSilenceSamples() const { return min_silence_samples_; }
    int SpeechPadLeftSamples() const { return speech_pad_left_samples_; }
    int SpeechPadRightSamples() const { return speech_pad_right_samples_; }

    std::string ModelName() const override;

  private:
    bool Initialize();

  private:
    std::mutex init_lock_;
    bool initialized_{false};

    // input and output
    std::vector<fastdeploy::FDTensor> inputTensors_;
    std::vector<fastdeploy::FDTensor> outputTensors_;

    // model states
    bool triggerd_ = false;
    unsigned int speech_start_ = 0;
    unsigned int speech_end_ = 0;
    unsigned int temp_end_ = 0;
    unsigned int current_sample_ = 0;
    unsigned int current_chunk_size_ = 0;
    // MAX 4294967295 samples / 8sample per ms / 1000 / 60 = 8947 minutes
    float outputProb_;

    std::vector<float> speakStart_;
    mutable std::vector<float> speakEnd_;

    std::vector<State> states_;

    /* ========================================================================
     */
    int sample_rate_ = 16000;
    int frame_ms_ = 32;  // 32, 64, 96 for 16k
    float threshold_ = 0.5f;

    int64_t window_size_samples_;  // support 256 512 768 for 8k; 512 1024 1536
                                   // for 16k.
    int sr_per_ms_;                // support 8 or 16
    int min_silence_samples_;      // sr_per_ms_ * frame_ms_
    int speech_pad_left_samples_{0};   // usually 250ms
    int speech_pad_right_samples_{0};  // usually 0

    /* ========================================================================
     */
    std::vector<int64_t> sr_;
    const size_t size_hc_ = 2 * 1 * 64;  // It's FIXED.
    std::vector<float> h_;
    std::vector<float> c_;

    std::vector<int64_t> input_node_dims_;
    const std::vector<int64_t> sr_node_dims_ = {1};
    const std::vector<int64_t> hc_node_dims_ = {2, 1, 64};
};

}  // namepsace ppspeech