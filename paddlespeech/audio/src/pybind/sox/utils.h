// Copyright (c) 2017 Facebook Inc. (Soumith Chintala),
// All rights reserved.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <sox.h>
#include "paddlespeech/audio/src/optional/optional.hpp"
#include "paddlespeech/audio/src/sox/utils.h"
#include "paddlespeech/audio/src/sox/types.h"

namespace py = pybind11;

namespace paddleaudio {
namespace sox_utils {

auto read_fileobj(py::object *fileobj, uint64_t size, char *buffer) -> uint64_t;

}  // namespace paddleaudio
}  // namespace sox_utils
