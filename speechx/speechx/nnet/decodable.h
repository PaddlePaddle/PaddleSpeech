#include "nnet/decodable-itf.h"
#include "base/common.h"
#include "kaldi/matrix/kaldi-matrix.h"
#include "frontend/feature_extractor_interface.h"
#include "nnet/nnet_interface.h"

namespace ppspeech {

struct DecodableOpts;

class Decodable : public kaldi::DecodableInterface {
  public: 
    explicit Decodable(const std::shared_ptr<NnetInterface>& nnet);
    //void Init(DecodableOpts config);
    virtual kaldi::BaseFloat LogLikelihood(int32 frame, int32 index);
    virtual bool IsLastFrame(int32 frame) const;
    virtual int32 NumIndices() const;
    virtual std::vector<BaseFloat> FrameLogLikelihood(int32 frame);
    void Acceptlikelihood(const kaldi::Matrix<kaldi::BaseFloat>& likelihood); // remove later
    void FeedFeatures(const kaldi::Matrix<kaldi::BaseFloat>& feature); // only for test, todo remove later
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
