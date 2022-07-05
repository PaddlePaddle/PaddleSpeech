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

#include "paddlespeech/audio/src/pybind/kaldi/kaldi_feature_wrapper.h"

namespace paddleaudio {
namespace kaldi {

KaldiFeatureWrapper* KaldiFeatureWrapper::GetInstance() {
    static KaldiFeatureWrapper instance;
    return &instance;
}

bool KaldiFeatureWrapper::InitFbank(::kaldi::FbankOptions opts) {
    fbank_.reset(new Fbank(opts));
    return true;
}

py::array_t<double> KaldiFeatureWrapper::ComputeFbank(
    const py::array_t<double> wav) {
    py::buffer_info info = wav.request();
    ::kaldi::Vector<::kaldi::BaseFloat> input_wav(info.size);
    double* wav_ptr = (double*)info.ptr;
    for (int idx = 0; idx < info.size; ++idx) {
        input_wav(idx) = *wav_ptr;
        wav_ptr++;
    }


    ::kaldi::Vector<::kaldi::BaseFloat> feats;
    bool flag = fbank_->ComputeFeature(input_wav, &feats);
    if (flag == false || feats.Dim() == 0) return py::array_t<double>();
    auto result = py::array_t<double>(feats.Dim());
    py::buffer_info xs = result.request();
    std::cout << std::endl;
    double* res_ptr = (double*)xs.ptr;
    for (int idx = 0; idx < feats.Dim(); ++idx) {
        *res_ptr = feats(idx);
        res_ptr++;
    }

    return result.reshape({feats.Dim() / Dim(), Dim()});
    /*
         py::buffer_info info = wav.request();
        std::cout << info.size << std::endl;
        auto result = py::array_t<double>(info.size);
       //::kaldi::Vector<::kaldi::BaseFloat> input_wav(info.size);
        ::kaldi::Vector<double> input_wav(info.size);
        py::buffer_info info_re = result.request();

        memcpy(input_wav.Data(), (double*)info.ptr, wav.nbytes());
        memcpy((double*)info_re.ptr, input_wav.Data(), input_wav.Dim()*
       sizeof(double));
        return result;
    */
}

}  // namesapce kaldi
}  // namespace paddleaudio
