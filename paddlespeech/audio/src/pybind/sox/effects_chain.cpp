#include <sox.h>
#include <iostream>
#include <vector>
#include "paddlespeech/audio/src/pybind/sox/effects_chain.h"
#include "paddlespeech/audio/src/pybind/sox/utils.h"

using namespace paddleaudio::sox_utils;

namespace paddleaudio::sox_effects_chain {

namespace {

/// helper classes for passing the location of input tensor and output buffer
///
/// drain/flow callback functions require plaing C style function signature and
/// the way to pass extra data is to attach data to sox_effect_t::priv pointer.
/// The following structs will be assigned to sox_effect_t::priv pointer which
/// gives sox_effect_t an access to input Tensor and output buffer object.
struct TensorInputPriv {
  size_t index;
  py::array* waveform;
  int64_t sample_rate;
  bool channels_first;
};

struct TensorOutputPriv {
  std::vector<sox_sample_t>* buffer;
};
struct FileOutputPriv {
  sox_format_t* sf;
};

/// Callback function to feed Tensor data to SoxEffectChain.
int tensor_input_drain(sox_effect_t* effp, sox_sample_t* obuf, size_t* osamp) {
  // Retrieve the input Tensor and current index
  auto priv = static_cast<TensorInputPriv*>(effp->priv);
  auto index = priv->index;
  auto tensor = *(priv->waveform);
  auto num_channels = effp->out_signal.channels;

  // Adjust the number of samples to read
  const size_t num_samples = tensor.size();
  if (index + *osamp > num_samples) {
    *osamp = num_samples - index;
  }

  // Ensure that it's a multiple of the number of channels
  *osamp -= *osamp % num_channels;

  // Slice the input Tensor
  // refacor this module, chunk
  auto i_frame = index / num_channels;
  auto num_frames = *osamp / num_channels;

  std::vector<int> chunk(num_frames*num_channels);
  py::buffer_info ori_info = tensor.request();
  void* ptr = ori_info.ptr;
  // Convert to sox_sample_t (int32_t)
  switch (tensor.dtype().num()) {
    //case c10::ScalarType::Float: {
    case 11: {
      // Need to convert to 64-bit precision so that
      // values around INT32_MIN/MAX are handled correctly.
      for (int idx = 0; idx < chunk.size(); ++idx) {
        int frame_idx = (idx + index) / num_channels;
        int channels_idx = (idx + index) % num_channels;
        double elem = 0; 
        if (priv->channels_first) {
          elem = *(float*)tensor.data(channels_idx, frame_idx);
        } else {
          elem = *(float*)tensor.data(frame_idx, channels_idx);
        } 
        elem = elem * 2147483648.;
        // *new_ptr = std::clamp(elem, INT32_MIN, INT32_MAX);
        if (elem > INT32_MAX) { 
          chunk[idx] = INT32_MAX; 
        } else if (elem < INT32_MIN) {
          chunk[idx] = INT32_MIN; 
        } else { 
          chunk[idx] = elem;
        }
      }
      break;
    }
    //case c10::ScalarType::Int: {
    case 5: {
      for (int idx = 0; idx < chunk.size(); ++idx) {
        int frame_idx = (idx + index) / num_channels;
        int channels_idx = (idx + index) % num_channels;
        int elem = 0;
        if (priv->channels_first) {
          elem = *(int*)tensor.data(channels_idx, frame_idx);
        } else {
          elem = *(int*)tensor.data(frame_idx, channels_idx);
        }
        chunk[idx] = elem;
      }
      break;
    }
    // case short
    case 3: {
      for (int idx = 0; idx < chunk.size(); ++idx) {
        int frame_idx = (idx + index) / num_channels;
        int channels_idx = (idx + index) % num_channels;
        int16_t elem = 0;
        if (priv->channels_first) {
          elem = *(int16_t*)tensor.data(channels_idx, frame_idx);
        } else {
          elem = *(int16_t*)tensor.data(frame_idx, channels_idx);
        }
        chunk[idx] = elem * 65536;
      }
      break;
    }
    // case byte
    case 1: {
      for (int idx = 0; idx < chunk.size(); ++idx) {
        int frame_idx = (idx + index) / num_channels;
        int channels_idx = (idx + index) % num_channels;
        int8_t elem = 0;
        if (priv->channels_first) {
          elem = *(int8_t*)tensor.data(channels_idx, frame_idx);
        } else {
          elem = *(int8_t*)tensor.data(frame_idx, channels_idx);
        }
        chunk[idx] = (elem - 128) * 16777216; 
      }
      break;
    }
    default:
      throw std::runtime_error("Unexpected dtype.");
  }
  // Write to buffer
  memcpy(obuf, chunk.data(), *osamp * 4);
  priv->index += *osamp;
  return (priv->index == num_samples) ? SOX_EOF : SOX_SUCCESS;
}

/// Callback function to fetch data from SoxEffectChain.
int tensor_output_flow(
    sox_effect_t* effp,
    sox_sample_t const* ibuf,
    sox_sample_t* obuf LSX_UNUSED,
    size_t* isamp,
    size_t* osamp) {
  *osamp = 0;
  // Get output buffer
  auto out_buffer = static_cast<TensorOutputPriv*>(effp->priv)->buffer;
  // Append at the end
  out_buffer->insert(out_buffer->end(), ibuf, ibuf + *isamp);
  return SOX_SUCCESS;
}

int file_output_flow(
    sox_effect_t* effp,
    sox_sample_t const* ibuf,
    sox_sample_t* obuf LSX_UNUSED,
    size_t* isamp,
    size_t* osamp) {
  *osamp = 0;
  if (*isamp) {
    auto sf = static_cast<FileOutputPriv*>(effp->priv)->sf;
    if (sox_write(sf, ibuf, *isamp) != *isamp) {
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

sox_effect_handler_t* get_tensor_input_handler() {
  static sox_effect_handler_t handler{
      /*name=*/"input_tensor",
      /*usage=*/NULL,
      /*flags=*/SOX_EFF_MCHAN,
      /*getopts=*/NULL,
      /*start=*/NULL,
      /*flow=*/NULL,
      /*drain=*/tensor_input_drain,
      /*stop=*/NULL,
      /*kill=*/NULL,
      /*priv_size=*/sizeof(TensorInputPriv)};
  return &handler;
}

sox_effect_handler_t* get_tensor_output_handler() {
  static sox_effect_handler_t handler{
      /*name=*/"output_tensor",
      /*usage=*/NULL,
      /*flags=*/SOX_EFF_MCHAN,
      /*getopts=*/NULL,
      /*start=*/NULL,
      /*flow=*/tensor_output_flow,
      /*drain=*/NULL,
      /*stop=*/NULL,
      /*kill=*/NULL,
      /*priv_size=*/sizeof(TensorOutputPriv)};
  return &handler;
}

sox_effect_handler_t* get_file_output_handler() {
  static sox_effect_handler_t handler{
      /*name=*/"output_file",
      /*usage=*/NULL,
      /*flags=*/SOX_EFF_MCHAN,
      /*getopts=*/NULL,
      /*start=*/NULL,
      /*flow=*/file_output_flow,
      /*drain=*/NULL,
      /*stop=*/NULL,
      /*kill=*/NULL,
      /*priv_size=*/sizeof(FileOutputPriv)};
  return &handler;
}

} // namespace

SoxEffect::SoxEffect(sox_effect_t* se) noexcept : se_(se) {}

SoxEffect::~SoxEffect() {
  if (se_ != nullptr) {
    free(se_);
  }
}

SoxEffect::operator sox_effect_t*() const {
  return se_;
}

auto SoxEffect::operator->() noexcept -> sox_effect_t* {
  return se_;
}

SoxEffectsChain::SoxEffectsChain(
    sox_encodinginfo_t input_encoding,
    sox_encodinginfo_t output_encoding)
    : in_enc_(input_encoding),
      out_enc_(output_encoding),
      in_sig_(),
      interm_sig_(),
      out_sig_(),
      sec_(sox_create_effects_chain(&in_enc_, &out_enc_)) {
  if (!sec_) {
    throw std::runtime_error("Failed to create effect chain.");
  }
}

SoxEffectsChain::~SoxEffectsChain() {
  if (sec_ != nullptr) {
    sox_delete_effects_chain(sec_);
  }
}

void SoxEffectsChain::run() {
  sox_flow_effects(sec_, NULL, NULL);
}

void SoxEffectsChain::addInputTensor(
    py::array* waveform,
    int64_t sample_rate,
    bool channels_first) {
  in_sig_ = get_signalinfo(waveform, sample_rate, "wav", channels_first);
  interm_sig_ = in_sig_;
  SoxEffect e(sox_create_effect(get_tensor_input_handler()));
  auto priv = static_cast<TensorInputPriv*>(e->priv);
  priv->index = 0;
  priv->waveform = waveform;
  priv->sample_rate = sample_rate;
  priv->channels_first = channels_first;
  if (sox_add_effect(sec_, e, &interm_sig_, &in_sig_) != SOX_SUCCESS) {
    throw std::runtime_error(
        "Internal Error: Failed to add effect: input_tensor");
  }
}

void SoxEffectsChain::addOutputBuffer(
    std::vector<sox_sample_t>* output_buffer) {
  SoxEffect e(sox_create_effect(get_tensor_output_handler()));
  static_cast<TensorOutputPriv*>(e->priv)->buffer = output_buffer;
  if (sox_add_effect(sec_, e, &interm_sig_, &in_sig_) != SOX_SUCCESS) {
    throw std::runtime_error(
        "Internal Error: Failed to add effect: output_tensor");
  }
}

void SoxEffectsChain::addInputFile(sox_format_t* sf) {
  in_sig_ = sf->signal;
  interm_sig_ = in_sig_;
  SoxEffect e(sox_create_effect(sox_find_effect("input")));
  char* opts[] = {(char*)sf};
  sox_effect_options(e, 1, opts);
  if (sox_add_effect(sec_, e, &interm_sig_, &in_sig_) != SOX_SUCCESS) {
    std::ostringstream stream;
    stream << "Internal Error: Failed to add effect: input " << sf->filename;
    throw std::runtime_error(stream.str());
  }
}

void SoxEffectsChain::addOutputFile(sox_format_t* sf) {
  out_sig_ = sf->signal;
  SoxEffect e(sox_create_effect(get_file_output_handler()));
  static_cast<FileOutputPriv*>(e->priv)->sf = sf;
  if (sox_add_effect(sec_, e, &interm_sig_, &out_sig_) != SOX_SUCCESS) {
    std::ostringstream stream;
    stream << "Internal Error: Failed to add effect: output " << sf->filename;
    throw std::runtime_error(stream.str());
  }
}

void SoxEffectsChain::addEffect(const std::vector<std::string> effect) {
  const auto num_args = effect.size();
  if (num_args == 0) {
    throw std::runtime_error("Invalid argument: empty effect.");
  }
  const auto name = effect[0];
  if (UNSUPPORTED_EFFECTS.find(name) != UNSUPPORTED_EFFECTS.end()) {
    std::ostringstream stream;
    stream << "Unsupported effect: " << name;
    throw std::runtime_error(stream.str());
  }

  auto returned_effect = sox_find_effect(name.c_str());
  if (!returned_effect) {
    std::ostringstream stream;
    stream << "Unsupported effect: " << name;
    throw std::runtime_error(stream.str());
  }
  SoxEffect e(sox_create_effect(returned_effect));
  const auto num_options = num_args - 1;

  std::vector<char*> opts;
  for (size_t i = 1; i < num_args; ++i) {
    opts.push_back((char*)effect[i].c_str());
  }
  if (sox_effect_options(e, num_options, num_options ? opts.data() : nullptr) !=
      SOX_SUCCESS) {
    std::ostringstream stream;
    stream << "Invalid effect option:";
    for (const auto& v : effect) {
      stream << " " << v;
    }
    throw std::runtime_error(stream.str());
  }

  if (sox_add_effect(sec_, e, &interm_sig_, &in_sig_) != SOX_SUCCESS) {
    std::ostringstream stream;
    stream << "Internal Error: Failed to add effect: \"" << name;
    for (size_t i = 1; i < num_args; ++i) {
      stream << " " << effect[i];
    }
    stream << "\"";
    throw std::runtime_error(stream.str());
  }
}

int64_t SoxEffectsChain::getOutputNumChannels() {
  return interm_sig_.channels;
}

int64_t SoxEffectsChain::getOutputSampleRate() {
  return interm_sig_.rate;
}

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
