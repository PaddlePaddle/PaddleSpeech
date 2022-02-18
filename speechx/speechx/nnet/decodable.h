#include "nnet/decodable-itf.h"

#include "base/common.h"

namespace ppspeech {

struct DecodableConfig;

class Decodable : public kaldi::DecodableInterface {
  public: 
    virtual void Init(DecodableOpts config);
    virtual kaldi::BaseFloat LogLikelihood(int32 frame, int32 index);
    virtual bool IsLastFrame(int32 frame) const;
    virtual int32 NumIndices() const;
    void Acceptlikelihood(const kaldi::Matrix<kaldi::BaseFloat>& likelihood); // remove later
    void FeedFeatures(const kaldi::Matrix<kaldi::BaseFloat>& feature); // only for test, todo remove later
    std::vector<BaseFloat> FrameLogLikelihood(int32 frame);
    void Reset();
    void InputFinished() { finished_ = true; }
  private:
    std::shared_ptr<FeatureExtractorInterface> frontend_;
    std::shared_ptr<NnetInterface> nnet_;
    kaldi::Matrix<kaldi::BaseFloat> nnet_cache_;
    bool finished_;
    int32 frames_ready_;
};

}  // namespace ppspeech