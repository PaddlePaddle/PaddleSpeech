//code is from: https://github.com/pytorch/audio/blob/main/torchaudio/csrc/sox/utils.h

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <sox.h>

#include "paddlespeech/audio/src/optional/optional.hpp"

namespace py = pybind11;

namespace paddleaudio {
namespace sox_utils {

////////////////////////////////////////////////////////////////////////////////
// APIs for Python interaction
////////////////////////////////////////////////////////////////////////////////

/// Set sox global options
void set_seed(const int64_t seed);

void set_verbosity(const int64_t verbosity);

void set_use_threads(const bool use_threads);

void set_buffer_size(const int64_t buffer_size);

int64_t get_buffer_size();

std::vector<std::vector<std::string>> list_effects();

std::vector<std::string> list_read_formats();

std::vector<std::string> list_write_formats();

////////////////////////////////////////////////////////////////////////////////
// Utilities for sox_io / sox_effects implementations
////////////////////////////////////////////////////////////////////////////////

const std::unordered_set<std::string> UNSUPPORTED_EFFECTS =
    {"input", "output", "spectrogram", "noiseprof", "noisered", "splice"};

/// helper class to automatically close sox_format_t*
struct SoxFormat {
  explicit SoxFormat(sox_format_t* fd) noexcept;
  SoxFormat(const SoxFormat& other) = delete;
  SoxFormat(SoxFormat&& other) = delete;
  SoxFormat& operator=(const SoxFormat& other) = delete;
  SoxFormat& operator=(SoxFormat&& other) = delete;
  ~SoxFormat();
  sox_format_t* operator->() const noexcept;
  operator sox_format_t*() const noexcept;

  void close();

 private:
  sox_format_t* fd_;
};

///
/// Verify that input Tensor is 2D, CPU and either uin8, int16, int32 or float32
void validate_input_tensor(const py::array);

void validate_input_file(const SoxFormat& sf, const std::string& path);

void validate_input_memfile(const SoxFormat &sf);
///
/// Get target dtype for the given encoding and precision.
py::dtype get_dtype(
    const sox_encoding_t encoding,
    const unsigned precision);

///
/// Convert sox_sample_t buffer to uint8/int16/int32/float32 Tensor
/// NOTE: This function might modify the values in the input buffer to
/// reduce the number of memory copy.
/// @param buffer Pointer to buffer that contains audio data.
/// @param num_samples The number of samples to read.
/// @param num_channels The number of channels. Used to reshape the resulting
/// Tensor.
/// @param dtype Target dtype. Determines the output dtype and value range in
/// conjunction with normalization.
/// @param noramlize Perform normalization. Only effective when dtype is not
/// kFloat32. When effective, the output tensor is kFloat32 type and value range
/// is [-1.0, 1.0]
/// @param channels_first When True, output Tensor has shape of [num_channels,
/// num_frames].
py::array convert_to_tensor(
    sox_sample_t* buffer,
    const int32_t num_samples,
    const int32_t num_channels,
    const py::dtype dtype,
    const bool normalize,
    const bool channels_first);

/// Extract extension from file path
const std::string get_filetype(const std::string path);

/// Get sox_signalinfo_t for passing a py::array object.
sox_signalinfo_t get_signalinfo(
    const py::array* waveform,
    const int64_t sample_rate,
    const std::string filetype,
    const bool channels_first);

/// Get sox_encodinginfo_t for Tensor I/O
sox_encodinginfo_t get_tensor_encodinginfo(const py::dtype dtype);

/// Get sox_encodinginfo_t for saving to file/file object
sox_encodinginfo_t get_encodinginfo_for_save(
    const std::string& format,
    const py::dtype dtype,
    const tl::optional<double> compression,
    const tl::optional<std::string> encoding,
    const tl::optional<int64_t> bits_per_sample);


} // namespace sox_utils
} // namespace paddleaudio
