//code is from: https://github.com/pytorch/audio/blob/main/torchaudio/csrc/sox/effects.h
#pragma once

#include <pybind11/pybind11.h>
#include "paddlespeech/audio/src/sox/utils.h"

namespace py = pybind11;

namespace paddleaudio::sox_effects {

void initialize_sox_effects();

void shutdown_sox_effects();

auto apply_effects_tensor(
    py::array waveform,
    int64_t sample_rate,
    const std::vector<std::vector<std::string>>& effects,
    bool channels_first) -> std::tuple<py::array, int64_t>;

auto apply_effects_file(
    const std::string& path,
    const std::vector<std::vector<std::string>>& effects,
    tl::optional<bool> normalize,
    tl::optional<bool> channels_first,
    const tl::optional<std::string>& format)
    -> tl::optional<std::tuple<py::array, int64_t>>;

} // namespace torchaudio::sox_effects
