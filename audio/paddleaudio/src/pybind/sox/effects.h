// the code is from https://github.com/pytorch/audio/blob/main/torchaudio/csrc/sox/effects.h  with modification.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "paddleaudio/src/optional/optional.hpp"

namespace py = pybind11;

namespace paddleaudio::sox_effects {

auto apply_effects_fileobj(
    py::object fileobj,
    const std::vector<std::vector<std::string>>& effects,
    tl::optional<bool> normalize,
    tl::optional<bool> channels_first,
    tl::optional<std::string> format)
    -> tl::optional<std::tuple<py::array, int64_t>>;

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

} // namespace paddleaudio::sox_effects
