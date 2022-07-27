#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "paddlespeech/audio/src/optional/optional.hpp"

namespace py = pybind11;

namespace paddleaudio::sox_effects {

auto apply_effects_fileobj(
    py::object fileobj,
    const std::vector<std::vector<std::string>>& effects,
    tl::optional<bool> normalize,
    tl::optional<bool> channels_first,
    tl::optional<std::string> format)
    -> tl::optional<std::tuple<py::array, int64_t>>;

} // namespace paddleaudio::sox_effects
