#ifndef PADDLEAUDIO_PYBIND_SOX_IO_H
#define PADDLEAUDIO_PYBIND_SOX_IO_H

#include "pybind/sox/utils.h"

namespace paddleaudio {
namespace sox_io {

auto get_info_file(const std::string &path, const std::string &format)
    -> std::tuple<int64_t, int64_t, int64_t, int64_t, std::string>;

auto get_info_fileobj(py::object fileobj, const std::string &format)
    -> std::tuple<int64_t, int64_t, int64_t, int64_t, std::string>;

} // namespace paddleaudio
} // namespace sox_io

#endif
