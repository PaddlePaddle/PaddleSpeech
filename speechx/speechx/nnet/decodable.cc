#include "nnet/decodable.h"

namespace ppspeech {

Decodable::Acceptlikelihood(const kaldi::Matrix<BaseFloat>& likelihood) {
  frames_ready_ += likelihood.NumRows();
}

Decodable::Init(DecodableConfig config) {

}

Decodable::IsLastFrame(int32 frame) const {
  CHECK_LE(frame, frames_ready_);
  return finished_ && (frame == frames_ready_ - 1);
}

int32 Decodable::NumIndices() const {
  return 0;
}

void Decodable::LogLikelihood(int32 frame, int32 index) {
  return ;
}

void Decodable::FeedFeatures(const kaldi::Matrix<kaldi::BaseFloat>& features) {
  // skip frame ???
  nnet_->FeedForward(features, &nnet_cache_); 
  frames_ready_ += nnet_cache_.NumRows(); 
  return ;
}

void Decodable::Reset() {
  // frontend_.Reset();
  nnet_->Reset();
}

} // namespace ppspeech
