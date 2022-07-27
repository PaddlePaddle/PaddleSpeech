#pragma once

#include "paddlespeech/audio/src/sox/effects_chain.h"

namespace paddleaudio::sox_effects_chain {

class SoxEffectsChainPyBind : public SoxEffectsChain {
  using SoxEffectsChain::SoxEffectsChain;

 public:
  void addInputFileObj(
      sox_format_t* sf,
      char* buffer,
      uint64_t buffer_size,
      py::object* fileobj);

  void addOutputFileObj(
      sox_format_t* sf,
      char** buffer,
      size_t* buffer_size,
      py::object* fileobj);
};

} // namespace paddleaudio::sox_effects_chain

