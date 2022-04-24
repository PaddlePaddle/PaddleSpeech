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

#include <paddleaudio/csrc/pybind/sox/io.h>

PYBIND11_MODULE(_paddleaudio, m) {
  m.def("get_info_file", &paddleaudio::sox_io::get_info_file,
        "Get metadata of audio file.");
  m.def("get_info_fileobj", &paddleaudio::sox_io::get_info_fileobj,
        "Get metadata of audio in file object.");
}
