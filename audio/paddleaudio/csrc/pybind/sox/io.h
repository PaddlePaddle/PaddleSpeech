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

#ifndef PADDLEAUDIO_PYBIND_SOX_IO_H
#define PADDLEAUDIO_PYBIND_SOX_IO_H

#include <paddleaudio/csrc/pybind/sox/utils.h>

namespace paddleaudio {
namespace sox_io {

auto get_info_file(const std::string &path, const std::string &format)
    -> std::tuple<int64_t, int64_t, int64_t, int64_t, std::string>;

auto get_info_fileobj(py::object fileobj, const std::string &format)
    -> std::tuple<int64_t, int64_t, int64_t, int64_t, std::string>;

} // namespace paddleaudio
} // namespace sox_io

#endif
