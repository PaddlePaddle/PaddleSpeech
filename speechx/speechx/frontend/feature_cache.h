#include "frontend/feature_extractor_interface.h"

class FeatureCache {
  public:
    explicit FeatureCache(FeatureExtractorInterface base_extractor); 
    void AcceptWaveform(const kaldi::VectorBase<kaldi::BaseFloat>& input);
    void Read(kaldi::VectorBase<kaldi::BaseFloat>* feat);
    void Dim() { return base_extractor_->Dim(); }
    void SetFinished();
    bool IsFinished();

  private:
    bool finished_;
    mutable std::mutex mutex_;
    size_t max_size;
    std::queue<kaldi::Vector<BaseFloat>> cache_;
    std::shared_ptr<FeatureExtractorInterface> base_extractor_;
    std::condition_variable ready_feed_condition_;
    std::condition_variable ready_read_condition_;
    DISALLOW_COPY_AND_ASSGIN(FeatureCache);
};