//code is from: https://github.com/pytorch/audio/blob/main/torchaudio/csrc/sox/types.cpp

#include "paddlespeech/audio/src/sox/types.h"
#include <ostream>
#include <sstream>

namespace paddleaudio {
namespace sox_utils {

Format get_format_from_string(const std::string& format) {
  if (format == "wav")
    return Format::WAV;
  if (format == "mp3")
    return Format::MP3;
  if (format == "flac")
    return Format::FLAC;
  if (format == "ogg" || format == "vorbis")
    return Format::VORBIS;
  if (format == "amr-nb")
    return Format::AMR_NB;
  if (format == "amr-wb")
    return Format::AMR_WB;
  if (format == "amb")
    return Format::AMB;
  if (format == "sph")
    return Format::SPHERE;
  if (format == "htk")
    return Format::HTK;
  if (format == "gsm")
    return Format::GSM;
  std::ostringstream stream;
  stream << "Internal Error: unexpected format value: " << format;
  throw std::runtime_error(stream.str());
}

std::string to_string(Encoding v) {
  switch (v) {
    case Encoding::UNKNOWN:
      return "UNKNOWN";
    case Encoding::PCM_SIGNED:
      return "PCM_S";
    case Encoding::PCM_UNSIGNED:
      return "PCM_U";
    case Encoding::PCM_FLOAT:
      return "PCM_F";
    case Encoding::FLAC:
      return "FLAC";
    case Encoding::ULAW:
      return "ULAW";
    case Encoding::ALAW:
      return "ALAW";
    case Encoding::MP3:
      return "MP3";
    case Encoding::VORBIS:
      return "VORBIS";
    case Encoding::AMR_WB:
      return "AMR_WB";
    case Encoding::AMR_NB:
      return "AMR_NB";
    case Encoding::OPUS:
      return "OPUS";
    default:
      throw std::runtime_error("Internal Error: unexpected encoding.");
  }
}

Encoding get_encoding_from_option(const tl::optional<std::string> encoding) {
  if (!encoding.has_value())
    return Encoding::NOT_PROVIDED;
  std::string v = encoding.value();
  if (v == "PCM_S")
    return Encoding::PCM_SIGNED;
  if (v == "PCM_U")
    return Encoding::PCM_UNSIGNED;
  if (v == "PCM_F")
    return Encoding::PCM_FLOAT;
  if (v == "ULAW")
    return Encoding::ULAW;
  if (v == "ALAW")
    return Encoding::ALAW;
  std::ostringstream stream;
  stream << "Internal Error: unexpected encoding value: " << v;
  throw std::runtime_error(stream.str());
}

BitDepth get_bit_depth_from_option(const tl::optional<int64_t> bit_depth) {
  if (!bit_depth.has_value())
    return BitDepth::NOT_PROVIDED;
  int64_t v = bit_depth.value();
  switch (v) {
    case 8:
      return BitDepth::B8;
    case 16:
      return BitDepth::B16;
    case 24:
      return BitDepth::B24;
    case 32:
      return BitDepth::B32;
    case 64:
      return BitDepth::B64;
    default: {
      std::ostringstream s;
      s << "Internal Error: unexpected bit depth value: " << v;
      throw std::runtime_error(s.str());
    }
  }
}

std::string get_encoding(sox_encoding_t encoding) {
  switch (encoding) {
    case SOX_ENCODING_UNKNOWN:
      return "UNKNOWN";
    case SOX_ENCODING_SIGN2:
      return "PCM_S";
    case SOX_ENCODING_UNSIGNED:
      return "PCM_U";
    case SOX_ENCODING_FLOAT:
      return "PCM_F";
    case SOX_ENCODING_FLAC:
      return "FLAC";
    case SOX_ENCODING_ULAW:
      return "ULAW";
    case SOX_ENCODING_ALAW:
      return "ALAW";
    case SOX_ENCODING_MP3:
      return "MP3";
    case SOX_ENCODING_VORBIS:
      return "VORBIS";
    case SOX_ENCODING_AMR_WB:
      return "AMR_WB";
    case SOX_ENCODING_AMR_NB:
      return "AMR_NB";
    case SOX_ENCODING_OPUS:
      return "OPUS";
    case SOX_ENCODING_GSM:
      return "GSM";
    default:
      return "UNKNOWN";
  }
}

} // namespace sox_utils
} // namespace paddleaudio
