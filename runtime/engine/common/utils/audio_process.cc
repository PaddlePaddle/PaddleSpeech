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

#include "utils/audio_process.h"

namespace ppspeech{

int WaveformFloatNormal(std::vector<float>* waveform) {
    int tot_samples = waveform->size();
    for (int i = 0; i < tot_samples; i++) {
        (*waveform)[i] = (*waveform)[i] / 32768.0;
    }
    return 0;
}

int WaveformNormal(std::vector<float>* waveform,
                   bool wav_normal,
                   const std::string& wav_normal_type,
                   float wav_norm_mul_factor) {
    if (wav_normal == false) {
        return 0;
    }
    if (wav_normal_type == "linear") {
        float amax = INT32_MIN;
        for (int i = 0; i < waveform->size(); ++i) {
            float tmp = std::abs((*waveform)[i]);
            amax = std::max(amax, tmp);
        }
        float factor = 1.0 / (amax + 1e-8);
        for (int i = 0; i < waveform->size(); ++i) {
            (*waveform)[i] = (*waveform)[i] * factor * wav_norm_mul_factor;
        }
    } else if (wav_normal_type == "gaussian") {
        double sum = std::accumulate(waveform->begin(), waveform->end(), 0.0);
        double mean = sum / waveform->size();  //均值

        double accum = 0.0;
        std::for_each(waveform->begin(), waveform->end(), [&](const double d) {
            accum += (d - mean) * (d - mean);
        });

        double stdev = sqrt(accum / (waveform->size() - 1));  //方差
        stdev = std::max(stdev, 1e-8);

        for (int i = 0; i < waveform->size(); ++i) {
            (*waveform)[i] =
                wav_norm_mul_factor * ((*waveform)[i] - mean) / stdev;
        }
    } else {
        printf("don't support\n");
        return -1;
    }
    return 0;
}

float PowerTodb(float in, float ref_value, float amin, float top_db) {
    if (amin <= 0) {
        printf("amin must be strictly positive\n");
        return -1;
    }

    if (ref_value <= 0) {
        printf("ref_value must be strictly positive\n");
        return -1;
    }

    float out = 10.0 * log10(std::max(amin, in));
    out -= 10.0 * log10(std::max(ref_value, amin));
    return out;
}

} // namespace ppspeech