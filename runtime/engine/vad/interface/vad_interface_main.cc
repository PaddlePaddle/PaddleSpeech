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


#include <iostream>
#include <vector>

#include <fstream>
#include "common/base/common.h"
#include "vad/frontend/wav.h"
#include "vad/interface/vad_interface.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: vad_interface_main path/to/config wav.scp"
                     "run_option, "
                     "e.g ./vad_interface_main config wav.scp"
                  << std::endl;
        return -1;
    }

    std::string config_path = argv[1];
    std::string wav_scp = argv[2];

    PPSHandle_t handle = PPSVadCreateInstance(config_path.c_str());

    std::ifstream fp_wav(wav_scp);
    std::string line = "";
    while(getline(fp_wav, line)){
        std::vector<float> inputWav;  // [0, 1]
        wav::WavReader wav_reader = wav::WavReader(line);
        auto sr = wav_reader.sample_rate();
        CHECK(sr == 16000) << " sr is " << sr << " expect 16000";

        auto num_samples = wav_reader.num_samples();
        inputWav.resize(num_samples);
        for (int i = 0; i < num_samples; i++) {
            inputWav[i] = wav_reader.data()[i] / 32768;
        }

        ppspeech::Timer timer;
        int window_size_samples = PPSVadChunkSizeSamples(handle);
        for (int64_t j = 0; j < num_samples; j += window_size_samples) {
            auto start = j;
            auto end = start + window_size_samples >= num_samples
                        ? num_samples
                        : start + window_size_samples;
            std::vector<float> r(window_size_samples, 0);
            auto current_chunk_size = end - start;
            memcpy(r.data(), inputWav.data() + start, current_chunk_size * sizeof(float));

            PPSVadState_t s = PPSVadFeedForward(handle, r.data(), r.size());
        }

        std::cout << "RTF=" << timer.Elapsed() / double(num_samples / sr)
                << std::endl;

        char result[10240] = {0};
        PPSVadGetResult(handle, result, 10240);
        std::cout << line << " " << result << std::endl;
        
        PPSVadReset(handle);
        // getchar();
    }
    PPSVadDestroyInstance(handle);
    return 0;
}
