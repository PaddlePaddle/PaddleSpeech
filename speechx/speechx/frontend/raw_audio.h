
#pragma once

#include "frontend/feature_extractor_interface.h"
#include "base/common.h"

#pragma once

namespace ppspeech {

class RawAudioSource {
  public:
    RawAudioSource(int buffer_size = kint16max);
    virtual void AcceptWaveform(kaldi::BaseFloat* data, int length);
    void AcceptWaveformByByte(char* data, lnt length) {}
    void AcceptWaveformByShort(kaldi::int16* data, int length) {}

    // read chunk data in buffer
    bool Read(VectorBase<BaseFloat>* feats);
    void SetFinished() { finished_ = true; }
    bool IsFinished() { return finished_; }

  private:
    vector<kaldi::BaseFloat> ring_buffer_;
    size_t start_;
    size_t data_length_;
    bool finished_;
    mutable std::mutex mutext_;
    std::condition_variable ready_read_condition_;
    std::condition_variable ready_feed_condition_;
    kaldi::int32 timeout_;
};

} // namespace ppspeech