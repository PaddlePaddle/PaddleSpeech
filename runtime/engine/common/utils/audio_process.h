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

#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <math.h>

namespace ppspeech{
int WaveformFloatNormal(std::vector<float>* waveform);
int WaveformNormal(std::vector<float>* waveform,
                    bool wav_normal,
                    const std::string& wav_normal_type,
                    float wav_norm_mul_factor);
float PowerTodb(float in,
                float ref_value = 1.0,
                float amin = 1e-10,
                float top_db = 80.0);
} // namespace ppspeech