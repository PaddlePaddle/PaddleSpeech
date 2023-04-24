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

#include <front/front_interface.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <paddle_api.h>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include "Predictor.hpp"

using namespace paddle::lite_api;

DEFINE_string(
    sentence,
    "你好，欢迎使用语音合成服务",
    "Text to be synthesized (Chinese only. English will crash the program.)");
DEFINE_string(front_conf, "./front.conf", "Front configuration file");
DEFINE_string(acoustic_model,
              "./models/cpu/fastspeech2_csmsc_arm.nb",
              "Acoustic model .nb file");
DEFINE_string(vocoder,
              "./models/cpu/fastspeech2_csmsc_arm.nb",
              "vocoder .nb file");
DEFINE_string(output_wav, "./output/tts.wav", "Output WAV file");
DEFINE_string(wav_bit_depth,
              "16",
              "WAV bit depth, 16 (16-bit PCM) or 32 (32-bit IEEE float)");
DEFINE_string(wav_sample_rate,
              "24000",
              "WAV sample rate, should match the output of the vocoder");
DEFINE_string(cpu_thread, "1", "CPU thread numbers");

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    PredictorInterface *predictor;

    if (FLAGS_wav_bit_depth == "16") {
        predictor = new Predictor<int16_t>();
    } else if (FLAGS_wav_bit_depth == "32") {
        predictor = new Predictor<float>();
    } else {
        LOG(ERROR) << "Unsupported WAV bit depth: " << FLAGS_wav_bit_depth;
        return -1;
    }


    /////////////////////////// 前端：文本转音素 ///////////////////////////

    // 实例化文本前端引擎
    ppspeech::FrontEngineInterface *front_inst = nullptr;
    front_inst = new ppspeech::FrontEngineInterface(FLAGS_front_conf);
    if ((!front_inst) || (front_inst->init())) {
        LOG(ERROR) << "Creater tts engine failed!";
        if (front_inst != nullptr) {
            delete front_inst;
        }
        front_inst = nullptr;
        return -1;
    }

    std::wstring ws_sentence = ppspeech::utf8string2wstring(FLAGS_sentence);

    // 繁体转简体
    std::wstring sentence_simp;
    front_inst->Trand2Simp(ws_sentence, &sentence_simp);
    ws_sentence = sentence_simp;

    std::string s_sentence;
    std::vector<std::wstring> sentence_part;
    std::vector<int> phoneids = {};
    std::vector<int> toneids = {};

    // 根据标点进行分句
    LOG(INFO) << "Start to segment sentences by punctuation";
    front_inst->SplitByPunc(ws_sentence, &sentence_part);
    LOG(INFO) << "Segment sentences through punctuation successfully";

    // 分句后获取音素id
    LOG(INFO)
        << "Start to get the phoneme and tone id sequence of each sentence";
    for (int i = 0; i < sentence_part.size(); i++) {
        LOG(INFO) << "Raw sentence is: "
                  << ppspeech::wstring2utf8string(sentence_part[i]);
        front_inst->SentenceNormalize(&sentence_part[i]);
        s_sentence = ppspeech::wstring2utf8string(sentence_part[i]);
        LOG(INFO) << "After normalization sentence is: " << s_sentence;

        if (0 != front_inst->GetSentenceIds(s_sentence, &phoneids, &toneids)) {
            LOG(ERROR) << "TTS inst get sentence phoneids and toneids failed";
            return -1;
        }
    }
    LOG(INFO) << "The phoneids of the sentence is: "
              << limonp::Join(phoneids.begin(), phoneids.end(), " ");
    LOG(INFO) << "The toneids of the sentence is: "
              << limonp::Join(toneids.begin(), toneids.end(), " ");
    LOG(INFO) << "Get the phoneme id sequence of each sentence successfully";


    /////////////////////////// 后端：音素转音频 ///////////////////////////

    // WAV采样率（必须与模型输出匹配）
    // 如果播放速度和音调异常，请修改采样率
    // 常见采样率：16000, 24000, 32000, 44100, 48000, 96000
    const uint32_t wavSampleRate = std::stoul(FLAGS_wav_sample_rate);

    // CPU线程数
    const int cpuThreadNum = std::stol(FLAGS_cpu_thread);

    // CPU电源模式
    const PowerMode cpuPowerMode = PowerMode::LITE_POWER_HIGH;

    if (!predictor->Init(FLAGS_acoustic_model,
                         FLAGS_vocoder,
                         cpuPowerMode,
                         cpuThreadNum,
                         wavSampleRate)) {
        LOG(ERROR) << "predictor init failed" << std::endl;
        return -1;
    }

    std::vector<int64_t> phones(phoneids.size());
    std::transform(phoneids.begin(), phoneids.end(), phones.begin(), [](int x) {
        return static_cast<int64_t>(x);
    });

    if (!predictor->RunModel(phones)) {
        LOG(ERROR) << "predictor run model failed" << std::endl;
        return -1;
    }

    LOG(INFO) << "Inference time: " << predictor->GetInferenceTime() << " ms, "
              << "WAV size (without header): " << predictor->GetWavSize()
              << " bytes, "
              << "WAV duration: " << predictor->GetWavDuration() << " ms, "
              << "RTF: " << predictor->GetRTF() << std::endl;

    if (!predictor->WriteWavToFile(FLAGS_output_wav)) {
        LOG(ERROR) << "write wav file failed" << std::endl;
        return -1;
    }

    delete predictor;

    return 0;
}
