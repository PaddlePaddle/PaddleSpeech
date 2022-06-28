// #include "sox/effects.h"
// #include "sox/effects_chain.h"
#include "sox/io.h"
#include "sox/types.h"
#include "sox/utils.h"

using namespace torch::indexing;
using namespace paddleaudio::sox_utils;

namespace paddleaudio {
namespace sox_io {

tl::optional<MetaDataTuple> get_info_file(
    const std::string& path,
    const tl::optional<std::string>& format) {
  SoxFormat sf(sox_open_read(
      path.c_str(),
      /*signal=*/nullptr,
      /*encoding=*/nullptr,
      /*filetype=*/format.has_value() ? format.value().c_str() : nullptr));

  if (static_cast<sox_format_t*>(sf) == nullptr ||
      sf->encoding.encoding == SOX_ENCODING_UNKNOWN) {
    return {};
  }

  return std::forward_as_tuple(
      static_cast<int64_t>(sf->signal.rate),
      static_cast<int64_t>(sf->signal.length / sf->signal.channels),
      static_cast<int64_t>(sf->signal.channels),
      static_cast<int64_t>(sf->encoding.bits_per_sample),
      get_encoding(sf->encoding.encoding));
}

std::vector<std::vector<std::string>> get_effects(
    const tl::optional<int64_t>& frame_offset,
    const tl::optional<int64_t>& num_frames) {
  const auto offset = frame_offset.value_or(0);
  if (offset < 0) {
    throw std::runtime_error(
        "Invalid argument: frame_offset must be non-negative.");
  }
  const auto frames = num_frames.value_or(-1);
  if (frames == 0 || frames < -1) {
    throw std::runtime_error(
        "Invalid argument: num_frames must be -1 or greater than 0.");
  }

  std::vector<std::vector<std::string>> effects;
  if (frames != -1) {
    std::ostringstream os_offset, os_frames;
    os_offset << offset << "s";
    os_frames << "+" << frames << "s";
    effects.emplace_back(
        std::vector<std::string>{"trim", os_offset.str(), os_frames.str()});
  } else if (offset != 0) {
    std::ostringstream os_offset;
    os_offset << offset << "s";
    effects.emplace_back(std::vector<std::string>{"trim", os_offset.str()});
  }
  return effects;
}

tl::optional<std::tuple<torch::Tensor, int64_t>> load_audio_file(
    const std::string& path,
    const tl::optional<int64_t>& frame_offset,
    const tl::optional<int64_t>& num_frames,
    tl::optional<bool> normalize,
    tl::optional<bool> channels_first,
    const tl::optional<std::string>& format) {
  auto effects = get_effects(frame_offset, num_frames);
  return paddleaudio::sox_effects::apply_effects_file(
      path, effects, normalize, channels_first, format);
}

void save_audio_file(
    const std::string& path,
    torch::Tensor tensor,
    int64_t sample_rate,
    bool channels_first,
    tl::optional<double> compression,
    tl::optional<std::string> format,
    tl::optional<std::string> encoding,
    tl::optional<int64_t> bits_per_sample) {
  validate_input_tensor(tensor);

  const auto filetype = [&]() {
    if (format.has_value())
      return format.value();
    return get_filetype(path);
  }();

  if (filetype == "amr-nb") {
    const auto num_channels = tensor.size(channels_first ? 0 : 1);
    TORCH_CHECK(
        num_channels == 1, "amr-nb format only supports single channel audio.");
  } else if (filetype == "htk") {
    const auto num_channels = tensor.size(channels_first ? 0 : 1);
    TORCH_CHECK(
        num_channels == 1, "htk format only supports single channel audio.");
  } else if (filetype == "gsm") {
    const auto num_channels = tensor.size(channels_first ? 0 : 1);
    TORCH_CHECK(
        num_channels == 1, "gsm format only supports single channel audio.");
    TORCH_CHECK(
        sample_rate == 8000,
        "gsm format only supports a sampling rate of 8kHz.");
  }
  const auto signal_info =
      get_signalinfo(&tensor, sample_rate, filetype, channels_first);
  const auto encoding_info = get_encodinginfo_for_save(
      filetype, tensor.dtype(), compression, encoding, bits_per_sample);

  SoxFormat sf(sox_open_write(
      path.c_str(),
      &signal_info,
      &encoding_info,
      /*filetype=*/filetype.c_str(),
      /*oob=*/nullptr,
      /*overwrite_permitted=*/nullptr));

  if (static_cast<sox_format_t*>(sf) == nullptr) {
    throw std::runtime_error(
        "Error saving audio file: failed to open file " + path);
  }

  paddleaudio::sox_effects_chain::SoxEffectsChain chain(
      /*input_encoding=*/get_tensor_encodinginfo(tensor.dtype()),
      /*output_encoding=*/sf->encoding);
  chain.addInputTensor(&tensor, sample_rate, channels_first);
  chain.addOutputFile(sf);
  chain.run();
}

TORCH_LIBRARY_FRAGMENT(paddleaudio, m) {
  m.def("paddleaudio::sox_io_get_info", &paddleaudio::sox_io::get_info_file);
  m.def(
      "paddleaudio::sox_io_load_audio_file",
      &paddleaudio::sox_io::load_audio_file);
  m.def(
      "paddleaudio::sox_io_save_audio_file",
      &paddleaudio::sox_io::save_audio_file);
}

} // namespace sox_io
} // namespace paddleaudio