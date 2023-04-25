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


#include "common/base/common.h"
#include "vad/nnet/vad.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: vad_nnet_main path/to/model path/to/audio "
                     "run_option, "
                     "e.g ./vad_nnet_main silero_vad.onnx sample.wav"
                  << std::endl;
        return -1;
    }

    std::string model_file = argv[1];
    std::string audio_file = argv[2];

    int sr = 16000;
    ppspeech::Vad vad(model_file);
    // custom config, but must be set before init
    vad.SetConfig(sr, 32, 0.5f, 0.15, 200, 0, 0);
    vad.Init();

    std::vector<float> inputWav;  // [0, 1]
    wav::WavReader wav_reader = wav::WavReader(audio_file);
    assert(wav_reader.sample_rate() == sr);


    auto num_samples = wav_reader.num_samples();
    inputWav.resize(num_samples);
    for (int i = 0; i < num_samples; i++) {
        inputWav[i] = wav_reader.data()[i] / 32768;
    }

    ppspeech::Timer timer;
    int window_size_samples = vad.WindowSizeSamples();
    for (int64_t j = 0; j < num_samples; j += window_size_samples) {
        auto start = j;
        auto end = start + window_size_samples >= num_samples
                       ? num_samples
                       : start + window_size_samples;
        auto current_chunk_size = end - start;

        std::vector<float> r{&inputWav[0] + start, &inputWav[0] + end};
        assert(r.size() == static_cast<size_t>(current_chunk_size));

        if (!vad.ForwardChunk(r)) {
            std::cerr << "Failed to inference while using model:"
                      << vad.ModelName() << "." << std::endl;
            return false;
        }

        ppspeech::Vad::State s = vad.Postprocess();
        std::cout << s << " ";
    }
    std::cout << std::endl;

    std::cout << "RTF=" << timer.Elapsed() / double(num_samples / sr)
              << std::endl;
    std::cout << "\b\b " << std::endl;

    vad.Reset();

    return 0;
}
