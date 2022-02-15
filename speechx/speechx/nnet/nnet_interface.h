
#pragma once

#include "base/basic_types.h"
#include "kaldi/base/kaldi-types.h"

namespace ppspeech {

class NnetInterface {
  public:
    virtual ~NnetForwardInterface() {}
    virtual void FeedForward(const kaldi::Matrix<BaseFloat>& features, 
                             kaldi::Matrix<kaldi::BaseFloat>* inferences) const = 0;

};

}  // namespace ppspeech