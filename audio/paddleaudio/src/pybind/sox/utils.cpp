//code is from: https://github.com/pytorch/audio/blob/main/torchaudio/csrc/sox/utils.cpp with modification.
#include <sox.h>

#include "paddleaudio/src/pybind/sox/utils.h"
#include "paddleaudio/src/pybind/sox/types.h"

#include <sstream>

namespace paddleaudio {
namespace sox_utils {

auto read_fileobj(py::object *fileobj, const uint64_t size, char *buffer)
    -> uint64_t {
    uint64_t num_read = 0;
    while (num_read < size) {
        auto request = size - num_read;
        auto chunk = static_cast<std::string>(
            static_cast<py::bytes>(fileobj->attr("read")(request)));
        auto chunk_len = chunk.length();
        if (chunk_len == 0) {
            break;
        }
        if (chunk_len > request) {
            std::ostringstream message;
            message
                << "Requested up to " << request << " bytes but, "
                << "received " << chunk_len << " bytes. "
                << "The given object does not confirm to read protocol of file "
                   "object.";
            throw std::runtime_error(message.str());
        }
        memcpy(buffer, chunk.data(), chunk_len);
        buffer += chunk_len;
        num_read += chunk_len;
    }
    return num_read;
}


void set_seed(const int64_t seed) {
  sox_get_globals()->ranqd1 = static_cast<sox_int32_t>(seed);
}

void set_verbosity(const int64_t verbosity) {
  sox_get_globals()->verbosity = static_cast<unsigned>(verbosity);
}

void set_use_threads(const bool use_threads) {
  sox_get_globals()->use_threads = static_cast<sox_bool>(use_threads);
}

void set_buffer_size(const int64_t buffer_size) {
  sox_get_globals()->bufsiz = static_cast<size_t>(buffer_size);
}

int64_t get_buffer_size() {
  return sox_get_globals()->bufsiz;
}

std::vector<std::vector<std::string>> list_effects() {
  std::vector<std::vector<std::string>> effects;
  for (const sox_effect_fn_t* fns = sox_get_effect_fns(); *fns; ++fns) {
    const sox_effect_handler_t* handler = (*fns)();
    if (handler && handler->name) {
      if (UNSUPPORTED_EFFECTS.find(handler->name) ==
          UNSUPPORTED_EFFECTS.end()) {
        effects.emplace_back(std::vector<std::string>{
            handler->name,
            handler->usage ? std::string(handler->usage) : std::string("")});
      }
    }
  }
  return effects;
}

std::vector<std::string> list_write_formats() {
  std::vector<std::string> formats;
  for (const sox_format_tab_t* fns = sox_get_format_fns(); fns->fn; ++fns) {
    const sox_format_handler_t* handler = fns->fn();
    for (const char* const* names = handler->names; *names; ++names) {
      if (!strchr(*names, '/') && handler->write)
        formats.emplace_back(*names);
    }
  }
  return formats;
}

std::vector<std::string> list_read_formats() {
  std::vector<std::string> formats;
  for (const sox_format_tab_t* fns = sox_get_format_fns(); fns->fn; ++fns) {
    const sox_format_handler_t* handler = fns->fn();
    for (const char* const* names = handler->names; *names; ++names) {
      if (!strchr(*names, '/') && handler->read)
        formats.emplace_back(*names);
    }
  }
  return formats;
}

SoxFormat::SoxFormat(sox_format_t* fd) noexcept : fd_(fd) {}
SoxFormat::~SoxFormat() {
  close();
}

sox_format_t* SoxFormat::operator->() const noexcept {
  return fd_;
}
SoxFormat::operator sox_format_t*() const noexcept {
  return fd_;
}

void SoxFormat::close() {
  if (fd_ != nullptr) {
    sox_close(fd_);
    fd_ = nullptr;
  }
}

void validate_input_file(const SoxFormat& sf, const std::string& path) {
  if (static_cast<sox_format_t*>(sf) == nullptr) {
    throw std::runtime_error(
        "Error loading audio file: failed to open file " + path);
  }
  if (sf->encoding.encoding == SOX_ENCODING_UNKNOWN) {
    throw std::runtime_error("Error loading audio file: unknown encoding.");
  }
}

void validate_input_memfile(const SoxFormat &sf) {
    return validate_input_file(sf, "<in memory buffer>");
}

void validate_input_tensor(const py::array tensor) {
  if (tensor.ndim() != 2) {
    throw std::runtime_error("Input tensor has to be 2D.");
  }

  char dtype = tensor.dtype().char_();
  bool flag = (dtype == 'f') || (dtype == 'd') || (dtype == 'l') || (dtype == 'i');
  if (flag == false) {
      throw std::runtime_error(
          "Input tensor has to be one of float32, int32, int16 or uint8 type.");
  }
}

py::dtype get_dtype(
    const sox_encoding_t encoding,
    const unsigned precision) {
    switch (encoding) {
      case SOX_ENCODING_UNSIGNED: // 8-bit PCM WAV
        return py::dtype('u1');
      case SOX_ENCODING_SIGN2: // 16-bit, 24-bit, or 32-bit PCM WAV
        switch (precision) {
          case 16:
            return py::dtype("i2");
          case 24: // Cast 24-bit to 32-bit.
          case 32:
            return py::dtype('i');
          default:
            throw std::runtime_error(
                "Only 16, 24, and 32 bits are supported for signed PCM.");
        }
      default:
        // default to float32 for the other formats, including
        // 32-bit flaoting-point WAV,
        // MP3,
        // FLAC,
        // VORBIS etc...
        return py::dtype("f");
    }
}

py::array convert_to_tensor(
    sox_sample_t* buffer,
    const int32_t num_samples,
    const int32_t num_channels,
    const py::dtype dtype,
    const bool normalize,
    const bool channels_first) {
  // todo refector later(SGoat)
  py::array t;
  uint64_t dummy = 0;
  SOX_SAMPLE_LOCALS;
  int32_t num_rows = num_samples / num_channels;
  if (normalize || dtype.char_() == 'f') {
    t = py::array(dtype, {num_rows, num_channels});
    auto ptr = (float*)t.mutable_data(0, 0);
    for (int32_t i = 0; i < num_samples; ++i) {
      ptr[i] = SOX_SAMPLE_TO_FLOAT_32BIT(buffer[i], dummy);
    }
    if (channels_first) {
    py::array t2 = py::array(dtype, {num_channels, num_rows});
    for (int32_t row_idx = 0; row_idx < num_channels; ++row_idx) {
      for (int32_t col_idx = 0; col_idx < num_rows; ++col_idx)
       *(float*)t2.mutable_data(row_idx, col_idx) = *(float*)t.data(col_idx, row_idx);
    }
    return t2;
  }
  } else if (dtype.char_() == 'i') {
    t = py::array(dtype, {num_rows, num_channels});
    auto ptr = (int*)t.mutable_data(0, 0);
    for (int32_t i = 0; i < num_samples; ++i) {
      ptr[i] = buffer[i];
    }
    if (channels_first) {
      py::array t2 = py::array(dtype, {num_channels, num_rows});
      for (int32_t row_idx = 0; row_idx < num_channels; ++row_idx) {
        for (int32_t col_idx = 0; col_idx < num_rows; ++col_idx)
          *(int*)t2.mutable_data(row_idx, col_idx) = *(int*)t.data(col_idx, row_idx);
      }
      return t2;
    }
  } else if (dtype.char_() == 'h') { // int16
    t = py::array(dtype, {num_rows, num_channels});
    auto ptr = (int16_t*)t.mutable_data(0, 0);
    for (int32_t i = 0; i < num_samples; ++i) {
      ptr[i] = SOX_SAMPLE_TO_SIGNED_16BIT(buffer[i], dummy);
    }
    if (channels_first) {
      py::array t2 = py::array(dtype, {num_channels, num_rows});
      for (int32_t row_idx = 0; row_idx < num_channels; ++row_idx) {
        for (int32_t col_idx = 0; col_idx < num_rows; ++col_idx)
          *(int16_t*)t2.mutable_data(row_idx, col_idx) = *(int16_t*)t.data(col_idx, row_idx);
      }
      return t2;
    }
  } else if (dtype.char_() == 'b') {
    //t = torch::empty({num_samples / num_channels, num_channels}, torch::kUInt8);
    t = py::array(dtype, {num_rows, num_channels});
    auto ptr = (uint8_t*)t.mutable_data(0,0);
    for (int32_t i = 0; i < num_samples; ++i) {
      ptr[i] = SOX_SAMPLE_TO_UNSIGNED_8BIT(buffer[i], dummy);
    }
    if (channels_first) {
      py::array t2 = py::array(dtype, {num_channels, num_rows});
      for (int32_t row_idx = 0; row_idx < num_channels; ++row_idx) {
        for (int32_t col_idx = 0; col_idx < num_rows; ++col_idx)
        *(uint8_t*)t2.mutable_data(row_idx, col_idx) = *(uint8_t*)t.data(col_idx, row_idx);
      }
      return t2;
    }
  } else {
    throw std::runtime_error("Unsupported dtype.");
  }
  return t;
}

const std::string get_filetype(const std::string path) {
  std::string ext = path.substr(path.find_last_of(".") + 1);
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  return ext;
}

namespace {

std::tuple<sox_encoding_t, unsigned> get_save_encoding_for_wav(
    const std::string format,
    py::dtype dtype,
    const Encoding& encoding,
    const BitDepth& bits_per_sample) {
  switch (encoding) {
    case Encoding::NOT_PROVIDED:
      switch (bits_per_sample) {
        case BitDepth::NOT_PROVIDED:
          switch (dtype.num()) {
            case 11: // float32 numpy dtype num 
              return std::make_tuple<>(SOX_ENCODING_FLOAT, 32);
            case 5: // int numpy dtype num
              return std::make_tuple<>(SOX_ENCODING_SIGN2, 32);
            case 3: // int16 numpy
              return std::make_tuple<>(SOX_ENCODING_SIGN2, 16);
            case 1: // byte numpy
              return std::make_tuple<>(SOX_ENCODING_UNSIGNED, 8);
            default:
              throw std::runtime_error("Internal Error: Unexpected dtype.");
          }
        case BitDepth::B8:
          return std::make_tuple<>(SOX_ENCODING_UNSIGNED, 8);
        default:
          return std::make_tuple<>(
              SOX_ENCODING_SIGN2, static_cast<unsigned>(bits_per_sample));
      }
    case Encoding::PCM_SIGNED:
      switch (bits_per_sample) {
        case BitDepth::NOT_PROVIDED:
          return std::make_tuple<>(SOX_ENCODING_SIGN2, 32);
        case BitDepth::B8:
          throw std::runtime_error(
              format + " does not support 8-bit signed PCM encoding.");
        default:
          return std::make_tuple<>(
              SOX_ENCODING_SIGN2, static_cast<unsigned>(bits_per_sample));
      }
    case Encoding::PCM_UNSIGNED:
      switch (bits_per_sample) {
        case BitDepth::NOT_PROVIDED:
        case BitDepth::B8:
          return std::make_tuple<>(SOX_ENCODING_UNSIGNED, 8);
        default:
          throw std::runtime_error(
              format + " only supports 8-bit for unsigned PCM encoding.");
      }
    case Encoding::PCM_FLOAT:
      switch (bits_per_sample) {
        case BitDepth::NOT_PROVIDED:
        case BitDepth::B32:
          return std::make_tuple<>(SOX_ENCODING_FLOAT, 32);
        case BitDepth::B64:
          return std::make_tuple<>(SOX_ENCODING_FLOAT, 64);
        default:
          throw std::runtime_error(
              format +
              " only supports 32-bit or 64-bit for floating-point PCM encoding.");
      }
    case Encoding::ULAW:
      switch (bits_per_sample) {
        case BitDepth::NOT_PROVIDED:
        case BitDepth::B8:
          return std::make_tuple<>(SOX_ENCODING_ULAW, 8);
        default:
          throw std::runtime_error(
              format + " only supports 8-bit for mu-law encoding.");
      }
    case Encoding::ALAW:
      switch (bits_per_sample) {
        case BitDepth::NOT_PROVIDED:
        case BitDepth::B8:
          return std::make_tuple<>(SOX_ENCODING_ALAW, 8);
        default:
          throw std::runtime_error(
              format + " only supports 8-bit for a-law encoding.");
      }
    default:
      throw std::runtime_error(
          format + " does not support encoding: " + to_string(encoding));
  }
}

std::tuple<sox_encoding_t, unsigned> get_save_encoding(
    const std::string& format,
    const py::dtype dtype,
    const tl::optional<std::string> encoding,
    const tl::optional<int64_t> bits_per_sample) {
  const Format fmt = get_format_from_string(format);
  const Encoding enc = get_encoding_from_option(encoding);
  const BitDepth bps = get_bit_depth_from_option(bits_per_sample);

  switch (fmt) {
    case Format::WAV:
    case Format::AMB:
      return get_save_encoding_for_wav(format, dtype, enc, bps);
    case Format::MP3:
      if (enc != Encoding::NOT_PROVIDED)
        throw std::runtime_error("mp3 does not support `encoding` option.");
      if (bps != BitDepth::NOT_PROVIDED)
        throw std::runtime_error(
            "mp3 does not support `bits_per_sample` option.");
      return std::make_tuple<>(SOX_ENCODING_MP3, 16);
    case Format::HTK:
      if (enc != Encoding::NOT_PROVIDED)
        throw std::runtime_error("htk does not support `encoding` option.");
      if (bps != BitDepth::NOT_PROVIDED)
        throw std::runtime_error(
            "htk does not support `bits_per_sample` option.");
      return std::make_tuple<>(SOX_ENCODING_SIGN2, 16);
    case Format::VORBIS:
      if (enc != Encoding::NOT_PROVIDED)
        throw std::runtime_error("vorbis does not support `encoding` option.");
      if (bps != BitDepth::NOT_PROVIDED)
        throw std::runtime_error(
            "vorbis does not support `bits_per_sample` option.");
      return std::make_tuple<>(SOX_ENCODING_VORBIS, 16);
    case Format::AMR_NB:
      if (enc != Encoding::NOT_PROVIDED)
        throw std::runtime_error("amr-nb does not support `encoding` option.");
      if (bps != BitDepth::NOT_PROVIDED)
        throw std::runtime_error(
            "amr-nb does not support `bits_per_sample` option.");
      return std::make_tuple<>(SOX_ENCODING_AMR_NB, 16);
    case Format::FLAC:
      if (enc != Encoding::NOT_PROVIDED)
        throw std::runtime_error("flac does not support `encoding` option.");
      switch (bps) {
        case BitDepth::B32:
        case BitDepth::B64:
          throw std::runtime_error(
              "flac does not support `bits_per_sample` larger than 24.");
        default:
          return std::make_tuple<>(
              SOX_ENCODING_FLAC, static_cast<unsigned>(bps));
      }
    case Format::SPHERE:
      switch (enc) {
        case Encoding::NOT_PROVIDED:
        case Encoding::PCM_SIGNED:
          switch (bps) {
            case BitDepth::NOT_PROVIDED:
              return std::make_tuple<>(SOX_ENCODING_SIGN2, 32);
            default:
              return std::make_tuple<>(
                  SOX_ENCODING_SIGN2, static_cast<unsigned>(bps));
          }
        case Encoding::PCM_UNSIGNED:
          throw std::runtime_error(
              "sph does not support unsigned integer PCM.");
        case Encoding::PCM_FLOAT:
          throw std::runtime_error("sph does not support floating point PCM.");
        case Encoding::ULAW:
          switch (bps) {
            case BitDepth::NOT_PROVIDED:
            case BitDepth::B8:
              return std::make_tuple<>(SOX_ENCODING_ULAW, 8);
            default:
              throw std::runtime_error(
                  "sph only supports 8-bit for mu-law encoding.");
          }
        case Encoding::ALAW:
          switch (bps) {
            case BitDepth::NOT_PROVIDED:
            case BitDepth::B8:
              return std::make_tuple<>(SOX_ENCODING_ALAW, 8);
            default:
              return std::make_tuple<>(
                  SOX_ENCODING_ALAW, static_cast<unsigned>(bps));
          }
        default:
          throw std::runtime_error(
              "sph does not support encoding: " + encoding.value());
      }
    case Format::GSM:
      if (enc != Encoding::NOT_PROVIDED)
        throw std::runtime_error("gsm does not support `encoding` option.");
      if (bps != BitDepth::NOT_PROVIDED)
        throw std::runtime_error(
            "gsm does not support `bits_per_sample` option.");
      return std::make_tuple<>(SOX_ENCODING_GSM, 16);

    default:
      throw std::runtime_error("Unsupported format: " + format);
  }
}

unsigned get_precision(const std::string filetype, py::dtype dtype) {
  if (filetype == "mp3")
    return SOX_UNSPEC;
  if (filetype == "flac")
    return 24;
  if (filetype == "ogg" || filetype == "vorbis")
    return SOX_UNSPEC;
  if (filetype == "wav" || filetype == "amb") {
    switch (dtype.num()) {
      case 1: // byte in numpy dype num
        return 8;
      case 3: // short, in numpy dtype num
        return 16;
      case 5: // int, numpy dtype 
        return 32;
      case 11: // float, numpy dtype
        return 32;
      default:
        throw std::runtime_error("Unsupported dtype.");
    }
  }
  if (filetype == "sph")
    return 32;
  if (filetype == "amr-nb") {
    return 16;
  }
  if (filetype == "gsm") {
    return 16;
  }
  if (filetype == "htk") {
    return 16;
  }
  throw std::runtime_error("Unsupported file type: " + filetype);
}

} // namespace

sox_signalinfo_t get_signalinfo(
    const py::array* waveform,
    const int64_t sample_rate,
    const std::string filetype,
    const bool channels_first) {
  return sox_signalinfo_t{
      /*rate=*/static_cast<sox_rate_t>(sample_rate),
      /*channels=*/
      static_cast<unsigned>(waveform->shape(channels_first ? 0 : 1)),
      /*precision=*/get_precision(filetype, waveform->dtype()),
      /*length=*/static_cast<uint64_t>(waveform->size())};
}

sox_encodinginfo_t get_tensor_encodinginfo(py::dtype dtype) {
  sox_encoding_t encoding = [&]() {
    switch (dtype.num()) {
      case 1: // byte
        return SOX_ENCODING_UNSIGNED;
      case 3: // short
        return SOX_ENCODING_SIGN2;
      case 5: // int32
        return SOX_ENCODING_SIGN2;
      case 11: // float
        return SOX_ENCODING_FLOAT;
      default:
        throw std::runtime_error("Unsupported dtype.");
    }
  }();
  unsigned bits_per_sample = [&]() {
    switch (dtype.num()) {
      case 1: // byte
        return 8;
      case 3: //short
        return 16;
      case 5: // int32
        return 32;
      case 11: // float
        return 32;
      default:
        throw std::runtime_error("Unsupported dtype.");
    }
  }();
  return sox_encodinginfo_t{
      /*encoding=*/encoding,
      /*bits_per_sample=*/bits_per_sample,
      /*compression=*/HUGE_VAL,
      /*reverse_bytes=*/sox_option_default,
      /*reverse_nibbles=*/sox_option_default,
      /*reverse_bits=*/sox_option_default,
      /*opposite_endian=*/sox_false};
}

sox_encodinginfo_t get_encodinginfo_for_save(
    const std::string& format,
    const py::dtype dtype,
    const tl::optional<double> compression,
    const tl::optional<std::string> encoding,
    const tl::optional<int64_t> bits_per_sample) {
  auto enc = get_save_encoding(format, dtype, encoding, bits_per_sample);
  return sox_encodinginfo_t{
      /*encoding=*/std::get<0>(enc),
      /*bits_per_sample=*/std::get<1>(enc),
      /*compression=*/compression.value_or(HUGE_VAL),
      /*reverse_bytes=*/sox_option_default,
      /*reverse_nibbles=*/sox_option_default,
      /*reverse_bits=*/sox_option_default,
      /*opposite_endian=*/sox_false};
}

}  // namespace paddleaudio
}  // namespace sox_utils
