
#pragma once

#include "base/basic_types.h"
#include "kaldi/base/kaldi-types.h"
#include "kaldi/matrix/kaldi-matrix.h"

namespace ppspeech {

class NnetInterface {
  public:
    virtual void FeedForward(const kaldi::Matrix<kaldi::BaseFloat>& features,
                             kaldi::Matrix<kaldi::BaseFloat>* inferences)= 0; 
    virtual void Reset() = 0;
    virtual ~NnetInterface() {}

};

}  // namespace ppspeech
