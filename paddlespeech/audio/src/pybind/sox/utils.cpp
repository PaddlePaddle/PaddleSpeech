// Copyright (c) 2017 Facebook Inc. (Soumith Chintala),
// All rights reserved.

#include "paddlespeech/audio/src/pybind/sox/utils.h"

#include <sstream>

namespace paddleaudio {
namespace sox_utils {

SoxFormat::SoxFormat(sox_format_t *fd) noexcept : fd_(fd) {}
SoxFormat::~SoxFormat() { close(); }

sox_format_t *SoxFormat::operator->() const noexcept { return fd_; }
SoxFormat::operator sox_format_t *() const noexcept { return fd_; }

void SoxFormat::close() {
    if (fd_ != nullptr) {
        sox_close(fd_);
        fd_ = nullptr;
    }
}

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

int64_t get_buffer_size() { return sox_get_globals()->bufsiz; }

void validate_input_file(const SoxFormat &sf, const std::string &path) {
    if (static_cast<sox_format_t *>(sf) == nullptr) {
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

}  // namespace paddleaudio
}  // namespace sox_utils
