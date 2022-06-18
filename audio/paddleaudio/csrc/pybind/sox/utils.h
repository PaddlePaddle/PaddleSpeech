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

#ifndef PADDLEAUDIO_PYBIND_SOX_UTILS_H
#define PADDLEAUDIO_PYBIND_SOX_UTILS_H

#include <pybind11/pybind11.h>
#include <sox.h>

namespace py = pybind11;

namespace paddleaudio {
namespace sox_utils {

/// helper class to automatically close sox_format_t*
struct SoxFormat {
  explicit SoxFormat(sox_format_t *fd) noexcept;
  SoxFormat(const SoxFormat &other) = delete;
  SoxFormat(SoxFormat &&other) = delete;
  SoxFormat &operator=(const SoxFormat &other) = delete;
  SoxFormat &operator=(SoxFormat &&other) = delete;
  ~SoxFormat();
  sox_format_t *operator->() const noexcept;
  operator sox_format_t *() const noexcept;

  void close();

private:
  sox_format_t *fd_;
};

auto read_fileobj(py::object *fileobj, uint64_t size, char *buffer) -> uint64_t;

int64_t get_buffer_size();

void validate_input_file(const SoxFormat &sf, const std::string &path);

void validate_input_memfile(const SoxFormat &sf);

std::string get_encoding(sox_encoding_t encoding);

} // namespace paddleaudio
} // namespace sox_utils

#endif
