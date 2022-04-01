#pragma once

#include "base/common.h"
#include "frontend/audio/frontend_itf.h"
#include "kaldi/matrix/kaldi-matrix.h"
#include "kaldi/util/options-itf.h"

namespace ppspeech {

class CMVN : public FrontendInterface {
  public:
    explicit CMVN(std::string cmvn_file,
                  std::unique_ptr<FrontendInterface> base_extractor);
    virtual void Accept(const kaldi::VectorBase<kaldi::BaseFloat>& inputs);

    // the length of feats = feature_row * feature_dim,
    // the Matrix is squashed into Vector
    virtual bool Read(kaldi::Vector<kaldi::BaseFloat>* feats);
    // the dim_ is the feautre dim.
    virtual size_t Dim() const { return dim_; }
    virtual void SetFinished() { base_extractor_->SetFinished(); }
    virtual bool IsFinished() const { return base_extractor_->IsFinished(); }
    virtual void Reset() { base_extractor_->Reset(); }

  private:
    void Compute(kaldi::VectorBase<kaldi::BaseFloat>* feats) const;
    void ApplyCMVN(kaldi::MatrixBase<BaseFloat>* feats);
    kaldi::Matrix<double> stats_;
    std::unique_ptr<FrontendInterface> base_extractor_;
    size_t dim_;
    bool var_norm_;
};

}  // namespace ppspeech