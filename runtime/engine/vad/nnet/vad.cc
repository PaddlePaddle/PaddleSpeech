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
#include "vad/nnet/vad.h"

#include <cstring>
#include <iomanip>

#include "common/base/common.h"


namespace ppspeech {

Vad::Vad(const std::string& model_file,
         const fastdeploy::RuntimeOption&
             custom_option /* = fastdeploy::RuntimeOption() */) {
    valid_cpu_backends = {fastdeploy::Backend::ORT,
                          fastdeploy::Backend::OPENVINO};
    valid_gpu_backends = {fastdeploy::Backend::ORT, fastdeploy::Backend::TRT};

    runtime_option = custom_option;
    // ORT backend
    runtime_option.UseCpu();
    runtime_option.UseOrtBackend();
    runtime_option.model_format = fastdeploy::ModelFormat::ONNX;
    // grap opt level
    runtime_option.ort_option.graph_optimization_level = 99;
    // one-thread
    runtime_option.ort_option.intra_op_num_threads = 1;
    runtime_option.ort_option.inter_op_num_threads = 1;
    // model path
    runtime_option.model_file = model_file;
}

void Vad::Init() {
    std::lock_guard<std::mutex> lock(init_lock_);
    Initialize();
}

std::string Vad::ModelName() const { return "VAD"; }

void Vad::SetConfig(const VadNnetConf conf) {
    SetConfig(conf.sr,
              conf.frame_ms,
              conf.threshold,
              conf.beam,
              conf.min_silence_duration_ms,
              conf.speech_pad_left_ms,
              conf.speech_pad_right_ms);
}

void Vad::SetConfig(const int& sr,
                    const int& frame_ms,
                    const float& threshold,
                    const float& beam,
                    const int& min_silence_duration_ms,
                    const int& speech_pad_left_ms,
                    const int& speech_pad_right_ms) {
    if (initialized_) {
        fastdeploy::FDERROR << "SetConfig must be called before init"
                            << std::endl;
        throw std::runtime_error("SetConfig must be called before init");
    }
    sample_rate_ = sr;
    sr_per_ms_ = sr / 1000;
    threshold_ = threshold;
    beam_ = beam;
    frame_ms_ = frame_ms;
    min_silence_samples_ = min_silence_duration_ms * sr_per_ms_;
    speech_pad_left_samples_ = speech_pad_left_ms * sr_per_ms_;
    speech_pad_right_samples_ = speech_pad_right_ms * sr_per_ms_;

    // init chunk size
    window_size_samples_ = frame_ms * sr_per_ms_;
    current_chunk_size_ = window_size_samples_;

    fastdeploy::FDINFO << "sr=" << sr_per_ms_ << " threshold=" << threshold_
                       << " beam=" << beam_ << " frame_ms=" << frame_ms_
                       << " min_silence_duration_ms=" << min_silence_duration_ms
                       << " speech_pad_left_ms=" << speech_pad_left_ms
                       << " speech_pad_right_ms=" << speech_pad_right_ms;
}

void Vad::Reset() {
    std::memset(h_.data(), 0.0f, h_.size() * sizeof(float));
    std::memset(c_.data(), 0.0f, c_.size() * sizeof(float));

    triggerd_ = false;
    temp_end_ = 0;
    current_sample_ = 0;

    speechStart_.clear();
    speechEnd_.clear();

    states_.clear();
}

bool Vad::Initialize() {
    // input & output holder
    inputTensors_.resize(4);
    outputTensors_.resize(3);

    // input shape
    input_node_dims_.emplace_back(1);
    input_node_dims_.emplace_back(window_size_samples_);
    // sr buffer
    sr_.resize(1);
    sr_[0] = sample_rate_;
    // hidden state buffer
    h_.resize(size_hc_);
    c_.resize(size_hc_);

    Reset();


    // InitRuntime
    if (!InitRuntime()) {
        fastdeploy::FDERROR << "Failed to initialize fastdeploy backend."
                            << std::endl;
        return false;
    }

    initialized_ = true;


    fastdeploy::FDINFO << "init done.";
    return true;
}

bool Vad::ForwardChunk(std::vector<float>& chunk) {
    // last chunk may not be window_size_samples_
    input_node_dims_.back() = chunk.size();
    assert(window_size_samples_ >= chunk.size());
    current_chunk_size_ = chunk.size();

    inputTensors_[0].name = "input";
    inputTensors_[0].SetExternalData(
        input_node_dims_, fastdeploy::FDDataType::FP32, chunk.data());
    inputTensors_[1].name = "sr";
    inputTensors_[1].SetExternalData(
        sr_node_dims_, fastdeploy::FDDataType::INT64, sr_.data());
    inputTensors_[2].name = "h";
    inputTensors_[2].SetExternalData(
        hc_node_dims_, fastdeploy::FDDataType::FP32, h_.data());
    inputTensors_[3].name = "c";
    inputTensors_[3].SetExternalData(
        hc_node_dims_, fastdeploy::FDDataType::FP32, c_.data());

    if (!Infer(inputTensors_, &outputTensors_)) {
        return false;
    }

    // Push forward sample index
    current_sample_ += current_chunk_size_;
    return true;
}

const Vad::State& Vad::Postprocess() {
    // update prob, h, c
    outputProb_ = *(float*)outputTensors_[0].Data();
    auto* hn = static_cast<float*>(outputTensors_[1].MutableData());
    std::memcpy(h_.data(), hn, h_.size() * sizeof(float));
    auto* cn = static_cast<float*>(outputTensors_[2].MutableData());
    std::memcpy(c_.data(), cn, c_.size() * sizeof(float));

    if (outputProb_ < threshold_ && !triggerd_) {
        // 1. Silence
#ifdef PPS_DEBUG
        DLOG(INFO) << "{ silence: " << 1.0 * current_sample_ / sample_rate_
                   << " s; prob: " << outputProb_ << " }";
#endif
        states_.emplace_back(Vad::State::SIL);
    } else if (outputProb_ >= threshold_ && !triggerd_) {
        // 2. Start
        triggerd_ = true;
        speech_start_ =
            current_sample_ - current_chunk_size_ - speech_pad_left_samples_;
        speech_start_ = std::max(int(speech_start_), 0);
        float start_sec = 1.0 * speech_start_ / sample_rate_;
        speechStart_.emplace_back(start_sec);
#ifdef PPS_DEBUG
        DLOG(INFO) << "{ speech start: " << start_sec
                   << " s; prob: " << outputProb_ << " }";
#endif
        states_.emplace_back(Vad::State::START);
    } else if (outputProb_ >= threshold_ - beam_ && triggerd_) {
        // 3. Continue

        if (temp_end_ != 0) {
            // speech prob relaxation, speech continues again
#ifdef PPS_DEBUG
            DLOG(INFO)
                << "{ speech fake end(sil < min_silence_ms) to continue: "
                << 1.0 * current_sample_ / sample_rate_
                << " s; prob: " << outputProb_ << " }";
#endif
            temp_end_ = 0;
        } else {
            // speech prob relaxation, keep tracking speech
#ifdef PPS_DEBUG
            DLOG(INFO) << "{ speech continue: "
                       << 1.0 * current_sample_ / sample_rate_
                       << " s; prob: " << outputProb_ << " }";
#endif
        }

        states_.emplace_back(Vad::State::SPEECH);
    } else if (outputProb_ < threshold_ - beam_ && triggerd_) {
        // 4. End
        if (temp_end_ == 0) {
            temp_end_ = current_sample_;
        }

        // check possible speech end
        if (current_sample_ - temp_end_ < min_silence_samples_) {
            // a. silence < min_slience_samples, continue speaking
#ifdef PPS_DEBUG
            DLOG(INFO) << "{ speech fake end(sil < min_silence_ms): "
                       << 1.0 * current_sample_ / sample_rate_
                       << " s; prob: " << outputProb_ << " }";
#endif
            states_.emplace_back(Vad::State::SIL);
        } else {
            // b. silence >= min_slience_samples, end speaking
            speech_end_ = current_sample_ + speech_pad_right_samples_;
            temp_end_ = 0;
            triggerd_ = false;
            auto end_sec = 1.0 * speech_end_ / sample_rate_;
            speechEnd_.emplace_back(end_sec);
#ifdef PPS_DEBUG
            DLOG(INFO) << "{ speech end: " << end_sec
                       << " s; prob: " << outputProb_ << " }";
#endif
            states_.emplace_back(Vad::State::END);
        }
    }

    return states_.back();
}

std::string Vad::ConvertTime(float time_s) const{
    float seconds_tmp, minutes_tmp, hours_tmp;
    float seconds;
    int minutes, hours;
 
	//	计算小时
	hours_tmp = time_s / 60 / 60;  // 1
	hours = (int)hours_tmp;
 
	// 计算分钟
	minutes_tmp = time_s / 60;
	if (minutes_tmp >= 60) {
		minutes = minutes_tmp - 60 * (double)hours;
	}
	else {
		minutes = minutes_tmp;
	}
 
	// 计算秒数
	seconds_tmp = (60 * 60 * hours) + (60 * minutes);
	seconds = time_s - seconds_tmp;
 
	// 输出格式
    std::stringstream ss;
    ss << hours << ":" << minutes << ":" << seconds;
 
	return ss.str();
}

int Vad::GetResult(char* result, int max_len,
    float removeThreshold,
    float expandHeadThreshold,
    float expandTailThreshold,
    float mergeThreshold) const {
    float audioLength = 1.0 * current_sample_ / sample_rate_;
    if (speechStart_.empty() && speechEnd_.empty()) {
        return {};
    }
    if (speechEnd_.size() != speechStart_.size()) {
        // set the audio length as the last end
        speechEnd_.emplace_back(audioLength);
    }
    
    std::string json = "[";

    for (int i = 0; i < speechStart_.size(); ++i) {
        json += "{\"s\":\"" + ConvertTime(speechStart_[i]) + "\",\"e\":\"" + ConvertTime(speechEnd_[i]) + "\"},";
    }
    json.pop_back();
    json += "]";
    
    if(result != NULL){
        snprintf(result, max_len, "%s", json.c_str());
    } else {
        DLOG(INFO) << "result is NULL";
    }
    return 0;
}

std::ostream& operator<<(std::ostream& os, const Vad::State& s) {
    switch (s) {
        case Vad::State::SIL:
            os << "[SIL]";
            break;
        case Vad::State::START:
            os << "[STA]";
            break;
        case Vad::State::SPEECH:
            os << "[SPE]";
            break;
        case Vad::State::END:
            os << "[END]";
            break;
        default:
            // illegal state
            os << "[ILL]";
            break;
    }
    return os;
}

}  // namespace ppspeech