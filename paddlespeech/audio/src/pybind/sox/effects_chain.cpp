#include <sox.h>

#include "paddlespeech/audio/src/pybind/sox/effects_chain.h"
#include "paddlespeech/audio/src/pybind/sox/utils.h"

using namespace paddleaudio::sox_utils;

namespace paddleaudio::sox_effects_chain {

namespace {

/// helper classes for passing file-like object to SoxEffectChain
struct FileObjInputPriv {
  sox_format_t* sf;
  py::object* fileobj;
  bool eof_reached;
  char* buffer;
  uint64_t buffer_size;
};

struct FileObjOutputPriv {
  sox_format_t* sf;
  py::object* fileobj;
  char** buffer;
  size_t* buffer_size;
};

/// Callback function to feed byte string
/// https://github.com/dmkrepo/libsox/blob/b9dd1a86e71bbd62221904e3e59dfaa9e5e72046/src/sox.h#L1268-L1278
auto fileobj_input_drain(sox_effect_t* effp, sox_sample_t* obuf, size_t* osamp)
    -> int {
  auto priv = static_cast<FileObjInputPriv*>(effp->priv);
  auto sf = priv->sf;
  auto buffer = priv->buffer;

  // 1. Refresh the buffer
  //
  // NOTE:
  //   Since the underlying FILE* was opened with `fmemopen`, the only way
  //   libsox detect EOF is reaching the end of the buffer. (null byte won't
  //   help) Therefore we need to align the content at the end of buffer,
  //   otherwise, libsox will keep reading the content beyond intended length.
  //
  // Before:
  //
  //     |<-------consumed------>|<---remaining--->|
  //     |***********************|-----------------|
  //                             ^ ftell
  //
  // After:
  //
  //     |<-offset->|<---remaining--->|<-new data->|
  //     |**********|-----------------|++++++++++++|
  //                ^ ftell

  // NOTE:
  //   Do not use `sf->tell_off` here. Presumably, `tell_off` and `fseek` are
  //   supposed to be in sync, but there are cases (Vorbis) they are not
  //   in sync and `tell_off` has seemingly uninitialized value, which
  //   leads num_remain to be negative and cause segmentation fault
  //   in `memmove`.
  const auto tell = ftell((FILE*)sf->fp);
  if (tell < 0) {
    throw std::runtime_error("Internal Error: ftell failed.");
  }
  const auto num_consumed = static_cast<size_t>(tell);
  if (num_consumed > priv->buffer_size) {
    throw std::runtime_error("Internal Error: buffer overrun.");
  }

  const auto num_remain = priv->buffer_size - num_consumed;

  // 1.1. Fetch the data to see if there is data to fill the buffer
  size_t num_refill = 0;
  std::string chunk(num_consumed, '\0');
  if (num_consumed && !priv->eof_reached) {
    num_refill = read_fileobj(
        priv->fileobj, num_consumed, const_cast<char*>(chunk.data()));
    if (num_refill < num_consumed) {
      priv->eof_reached = true;
    }
  }
  const auto offset = num_consumed - num_refill;

  // 1.2. Move the unconsumed data towards the beginning of buffer.
  if (num_remain) {
    auto src = static_cast<void*>(buffer + num_consumed);
    auto dst = static_cast<void*>(buffer + offset);
    memmove(dst, src, num_remain);
  }

  // 1.3. Refill the remaining buffer.
  if (num_refill) {
    auto src = static_cast<void*>(const_cast<char*>(chunk.c_str()));
    auto dst = buffer + offset + num_remain;
    memcpy(dst, src, num_refill);
  }

  // 1.4. Set the file pointer to the new offset
  sf->tell_off = offset;
  fseek((FILE*)sf->fp, offset, SEEK_SET);

  // 2. Perform decoding operation
  // The following part is practically same as "input" effect
  // https://github.com/dmkrepo/libsox/blob/b9dd1a86e71bbd62221904e3e59dfaa9e5e72046/src/input.c#L30-L48

  // At this point, osamp represents the buffer size in bytes,
  // but sox_read expects the maximum number of samples ready to read.
  // Normally, this is fine, but in case when the samples are not 4-byte
  // aligned, (e.g. sample is 24bits), the resulting signal is not correct.
  // https://github.com/pytorch/audio/issues/2083
  if (sf->encoding.bits_per_sample > 0)
    *osamp /= (sf->encoding.bits_per_sample / 8);

  // Ensure that it's a multiple of the number of channels
  *osamp -= *osamp % effp->out_signal.channels;

  // Read up to *osamp samples into obuf;
  // store the actual number read back to *osamp
  *osamp = sox_read(sf, obuf, *osamp);

  // Decoding is finished when fileobject is exhausted and sox can no longer
  // decode a sample.
  return (priv->eof_reached && !*osamp) ? SOX_EOF : SOX_SUCCESS;
}

auto fileobj_output_flow(
    sox_effect_t* effp,
    sox_sample_t const* ibuf,
    sox_sample_t* obuf LSX_UNUSED,
    size_t* isamp,
    size_t* osamp) -> int {
  *osamp = 0;
  if (*isamp) {
    auto priv = static_cast<FileObjOutputPriv*>(effp->priv);
    auto sf = priv->sf;
    auto fp = static_cast<FILE*>(sf->fp);
    auto fileobj = priv->fileobj;
    auto buffer = priv->buffer;

    // Encode chunk
    auto num_samples_written = sox_write(sf, ibuf, *isamp);
    fflush(fp);

    // Copy the encoded chunk to python object.
    fileobj->attr("write")(py::bytes(*buffer, ftell(fp)));

    // Reset FILE*
    sf->tell_off = 0;
    fseek(fp, 0, SEEK_SET);

    if (num_samples_written != *isamp) {
      if (sf->sox_errno) {
        std::ostringstream stream;
        stream << sf->sox_errstr << " " << sox_strerror(sf->sox_errno) << " "
               << sf->filename;
        throw std::runtime_error(stream.str());
      }
      return SOX_EOF;
    }
  }
  return SOX_SUCCESS;
}

auto get_fileobj_input_handler() -> sox_effect_handler_t* {
  static sox_effect_handler_t handler{
      /*name=*/"input_fileobj_object",
      /*usage=*/nullptr,
      /*flags=*/SOX_EFF_MCHAN,
      /*getopts=*/nullptr,
      /*start=*/nullptr,
      /*flow=*/nullptr,
      /*drain=*/fileobj_input_drain,
      /*stop=*/nullptr,
      /*kill=*/nullptr,
      /*priv_size=*/sizeof(FileObjInputPriv)};
  return &handler;
}

auto get_fileobj_output_handler() -> sox_effect_handler_t* {
  static sox_effect_handler_t handler{
      /*name=*/"output_fileobj_object",
      /*usage=*/nullptr,
      /*flags=*/SOX_EFF_MCHAN,
      /*getopts=*/nullptr,
      /*start=*/nullptr,
      /*flow=*/fileobj_output_flow,
      /*drain=*/nullptr,
      /*stop=*/nullptr,
      /*kill=*/nullptr,
      /*priv_size=*/sizeof(FileObjOutputPriv)};
  return &handler;
}

} // namespace

void SoxEffectsChainPyBind::addInputFileObj(
    sox_format_t* sf,
    char* buffer,
    uint64_t buffer_size,
    py::object* fileobj) {
  in_sig_ = sf->signal;
  interm_sig_ = in_sig_;

  SoxEffect e(sox_create_effect(get_fileobj_input_handler()));
  auto priv = static_cast<FileObjInputPriv*>(e->priv);
  priv->sf = sf;
  priv->fileobj = fileobj;
  priv->eof_reached = false;
  priv->buffer = buffer;
  priv->buffer_size = buffer_size;
  if (sox_add_effect(sec_, e, &interm_sig_, &in_sig_) != SOX_SUCCESS) {
    throw std::runtime_error(
        "Internal Error: Failed to add effect: input fileobj");
  }
}

void SoxEffectsChainPyBind::addOutputFileObj(
    sox_format_t* sf,
    char** buffer,
    size_t* buffer_size,
    py::object* fileobj) {
  out_sig_ = sf->signal;
  SoxEffect e(sox_create_effect(get_fileobj_output_handler()));
  auto priv = static_cast<FileObjOutputPriv*>(e->priv);
  priv->sf = sf;
  priv->fileobj = fileobj;
  priv->buffer = buffer;
  priv->buffer_size = buffer_size;
  if (sox_add_effect(sec_, e, &interm_sig_, &out_sig_) != SOX_SUCCESS) {
    throw std::runtime_error(
        "Internal Error: Failed to add effect: output fileobj");
  }
}

} // namespace paddleaudio::sox_effects_chain
