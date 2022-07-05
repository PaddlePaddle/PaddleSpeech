// Copyright (c) 2017 Facebook Inc. (Soumith Chintala), All rights reserved.
// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

#include "paddlespeech/audio/src/pybind/kaldi/kaldi_feature.h"
#include "paddlespeech/audio/src/pybind/sox/io.h"

// Sox
PYBIND11_MODULE(_paddleaudio, m) {
    m.def("get_info_file",
          &paddleaudio::sox_io::get_info_file,
          "Get metadata of audio file.");
    m.def("get_info_fileobj",
          &paddleaudio::sox_io::get_info_fileobj,
          "Get metadata of audio in file object.");

    m.def("InitFbank", &paddleaudio::kaldi::InitFbank, "init fbank");
    m.def("ResetFbank", &paddleaudio::kaldi::ResetFbank, "reset fbank");
    m.def("ComputeFbank", &paddleaudio::kaldi::ComputeFbank, "compute fbank");
    m.def("ComputeFbankStreaming",
          &paddleaudio::kaldi::ComputeFbankStreaming,
          "compute fbank streaming");
    m.def("ComputeKaldiPitch", &paddleaudio::kaldi::ComputeKaldiPitch, "compute kaldi pitch");
}
