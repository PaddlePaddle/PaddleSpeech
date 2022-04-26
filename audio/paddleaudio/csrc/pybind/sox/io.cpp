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
#include <paddleaudio/csrc/pybind/sox/utils.h>

using namespace paddleaudio::sox_utils;

namespace paddleaudio {
namespace sox_io {

auto get_info_file(const std::string &path, const std::string &format)
    -> std::tuple<int64_t, int64_t, int64_t, int64_t, std::string> {
  SoxFormat sf(sox_open_read(path.data(),
                             /*signal=*/nullptr,
                             /*encoding=*/nullptr,
                             /*filetype=*/format.empty() ? nullptr : format.data()));
  

  validate_input_file(sf, path);

  return std::make_tuple(
      static_cast<int64_t>(sf->signal.rate),
      static_cast<int64_t>(sf->signal.length / sf->signal.channels),
      static_cast<int64_t>(sf->signal.channels),
      static_cast<int64_t>(sf->encoding.bits_per_sample),
      get_encoding(sf->encoding.encoding));
}

auto get_info_fileobj(py::object fileobj, const std::string &format)
    -> std::tuple<int64_t, int64_t, int64_t, int64_t, std::string> {
  const auto capacity = [&]() {
    const auto bufsiz = get_buffer_size();
    const int64_t kDefaultCapacityInBytes = 4096;
    return (bufsiz > kDefaultCapacityInBytes) ? bufsiz
                                              : kDefaultCapacityInBytes;
  }();
  std::string buffer(capacity, '\0');
  auto *buf = const_cast<char *>(buffer.data());
  auto num_read = read_fileobj(&fileobj, capacity, buf);
  // If the file is shorter than 256, then libsox cannot read the header.
  auto buf_size = (num_read > 256) ? num_read : 256;

  SoxFormat sf(sox_open_mem_read(buf, buf_size,
                                 /*signal=*/nullptr,
                                 /*encoding=*/nullptr,
                                 /*filetype=*/format.empty() ? nullptr : format.data()));

  // In case of streamed data, length can be 0
  validate_input_memfile(sf);

  return std::make_tuple(
      static_cast<int64_t>(sf->signal.rate),
      static_cast<int64_t>(sf->signal.length / sf->signal.channels),
      static_cast<int64_t>(sf->signal.channels),
      static_cast<int64_t>(sf->encoding.bits_per_sample),
      get_encoding(sf->encoding.encoding));
}

} // namespace paddleaudio
} // namespace sox_io
