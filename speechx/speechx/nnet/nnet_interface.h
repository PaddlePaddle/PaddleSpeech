
#pragma once

#include ""

namespace ppspeech {

class NnetForwardInterface {
  public:
    virtual ~NnetForwardInterface() {}
    virtual void FeedForward(const kaldi::Matrix<BaseFloat>& features, 
                             kaldi::Vector<kaldi::BaseFloat>* inference) const = 0;

};

}  // namespace ppspeech