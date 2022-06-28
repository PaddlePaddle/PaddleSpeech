//Copyright (c) 2017 Facebook Inc. (Soumith Chintala), 
//All rights reserved.

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
