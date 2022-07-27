// Copyright (c) 2017 Facebook Inc. (Soumith Chintala),
// All rights reserved.

#include "paddlespeech/audio/src/pybind/sox/io.h"
#include "paddlespeech/audio/src/pybind/sox/effects.h"
#include "paddlespeech/audio/src/pybind/sox/effects_chain.h"
#include "paddlespeech/audio/src/pybind/sox/utils.h"
#include "paddlespeech/audio/src/optional/optional.hpp"

#include "paddlespeech/audio/src/sox/io.h"
#include "paddlespeech/audio/src/sox/types.h"
#include "paddlespeech/audio/src/sox/utils.h"

using namespace paddleaudio::sox_utils;

namespace paddleaudio {
namespace sox_io {

auto get_info_file(const std::string &path, const std::string &format)
    -> std::tuple<int64_t, int64_t, int64_t, int64_t, std::string> {
    SoxFormat sf(
        sox_open_read(path.data(),
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

    SoxFormat sf(sox_open_mem_read(
        buf,
        buf_size,
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

tl::optional<std::tuple<py::array, int64_t>> load_audio_fileobj(
    py::object fileobj,
    const tl::optional<int64_t>& frame_offset,
    const tl::optional<int64_t>& num_frames,
    tl::optional<bool> normalize,
    tl::optional<bool> channels_first,
    const tl::optional<std::string>& format) {
  auto effects = get_effects(frame_offset, num_frames);
  return paddleaudio::sox_effects::apply_effects_fileobj(
      std::move(fileobj), effects, normalize, channels_first, std::move(format));
}

namespace {
// helper class to automatically release buffer, to be used by
// save_audio_fileobj
struct AutoReleaseBuffer {
  char* ptr;
  size_t size;

  AutoReleaseBuffer() : ptr(nullptr), size(0) {}
  AutoReleaseBuffer(const AutoReleaseBuffer& other) = delete;
  AutoReleaseBuffer(AutoReleaseBuffer&& other) = delete;
  auto operator=(const AutoReleaseBuffer& other) -> AutoReleaseBuffer& = delete;
  auto operator=(AutoReleaseBuffer&& other) -> AutoReleaseBuffer& = delete;
  ~AutoReleaseBuffer() {
    if (ptr) {
      free(ptr);
    }
  }
};

} // namespace

void save_audio_fileobj(
    py::object fileobj,
    py::array tensor,
    int64_t sample_rate,
    bool channels_first,
    tl::optional<double> compression,
    tl::optional<std::string> format,
    tl::optional<std::string> encoding,
    tl::optional<int64_t> bits_per_sample) {

  if (!format.has_value()) {
    throw std::runtime_error(
        "`format` is required when saving to file object.");
  }
  const auto filetype = format.value();

  if (filetype == "amr-nb") {
    const auto num_channels = tensor.shape(channels_first ? 0 : 1);
    if (num_channels != 1) {
      throw std::runtime_error(
          "amr-nb format only supports single channel audio.");
    }
  } else if (filetype == "htk") {
    const auto num_channels = tensor.shape(channels_first ? 0 : 1);
    if (num_channels != 1) {
      throw std::runtime_error(
          "htk format only supports single channel audio.");
    }
  } else if (filetype == "gsm") {
    const auto num_channels = tensor.shape(channels_first ? 0 : 1);
    if (num_channels != 1) {
      throw std::runtime_error(
          "gsm format only supports single channel audio.");
    }
    if (sample_rate != 8000) {
      throw std::runtime_error(
          "gsm format only supports a sampling rate of 8kHz.");
    }
  }

  const auto signal_info =
      get_signalinfo(&tensor, sample_rate, filetype, channels_first);
  const auto encoding_info = get_encodinginfo_for_save(
      filetype,
      tensor.dtype(),
      compression,
      std::move(encoding),
      bits_per_sample);

  AutoReleaseBuffer buffer;

  SoxFormat sf(sox_open_memstream_write(
      &buffer.ptr,
      &buffer.size,
      &signal_info,
      &encoding_info,
      filetype.c_str(),
      /*oob=*/nullptr));

  if (static_cast<sox_format_t*>(sf) == nullptr) {
    throw std::runtime_error(
        "Error saving audio file: failed to open memory stream.");
  }

  paddleaudio::sox_effects_chain::SoxEffectsChainPyBind chain(
      /*input_encoding=*/get_tensor_encodinginfo(tensor.dtype()),
      /*output_encoding=*/sf->encoding);
  chain.addInputTensor(&tensor, sample_rate, channels_first);
  chain.addOutputFileObj(sf, &buffer.ptr, &buffer.size, &fileobj);
  chain.run();

  // Closing the sox_format_t is necessary for flushing the last chunk to the
  // buffer
  sf.close();
  fileobj.attr("write")(py::bytes(buffer.ptr, buffer.size));
}

}  // namespace paddleaudio
}  // namespace sox_io
