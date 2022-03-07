
#pragma once

#include "base/common.h"
#include "frontend/feature_extractor_interface.h"
#include "kaldi/util/options-itf.h"
#include "kaldi/matrix/kaldi-matrix.h"

namespace ppspeech {

struct DecibelNormalizerOptions {
  float target_db;
  float max_gain_db;
  bool convert_int_float;
  DecibelNormalizerOptions() :
    target_db(-20),
    max_gain_db(300.0),
    convert_int_float(false) {}

    void Register(kaldi::OptionsItf* opts) {
      opts->Register("target-db", &target_db, "target db for db normalization");
      opts->Register("max-gain-db", &max_gain_db, "max gain db for db normalization");
      opts->Register("convert-int-float", &convert_int_float, "if convert int samples to float");
    }
};

class DecibelNormalizer : public FeatureExtractorInterface {
  public:
    explicit DecibelNormalizer(const DecibelNormalizerOptions& opts);
    virtual void AcceptWaveform(const kaldi::VectorBase<kaldi::BaseFloat>& input);
    virtual void Read(kaldi::VectorBase<kaldi::BaseFloat>* feat);
    virtual size_t Dim() const { return dim_; }
    bool Compute(const kaldi::VectorBase<kaldi::BaseFloat>& input,
                 kaldi::VectorBase<kaldi::BaseFloat>* feat) const;
  private:
    DecibelNormalizerOptions opts_;
    size_t dim_;
    std::unique_ptr<FeatureExtractorInterface> base_extractor_;
    kaldi::Vector<kaldi::BaseFloat> waveform_;
};


class CMVN : public FeatureExtractorInterface {
  public:
    explicit CMVN(std::string cmvn_file);
    virtual void AcceptWaveform(const kaldi::VectorBase<kaldi::BaseFloat>& input);
    virtual void Read(kaldi::VectorBase<kaldi::BaseFloat>* feat);
    virtual size_t Dim() const { return stats_.NumCols() - 1; }
    bool Compute(const kaldi::VectorBase<kaldi::BaseFloat>& input,
                 kaldi::VectorBase<kaldi::BaseFloat>* feat) const;
    // for test
    void ApplyCMVN(bool var_norm, kaldi::VectorBase<BaseFloat>* feats);
    void ApplyCMVNMatrix(bool var_norm, kaldi::MatrixBase<BaseFloat>* feats);
  private:
    kaldi::Matrix<double> stats_;
    std::shared_ptr<FeatureExtractorInterface> base_extractor_;
    size_t dim_;
    bool var_norm_;
};

}  // namespace ppspeech