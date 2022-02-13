
#pragma once

#include "base/common.h"
#include "frontend/feature_extractor_interface.h"
#include "kaldi/util/options-itf.h"

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
    virtual size_t Dim() const { return 0; }
    bool Compute(const kaldi::VectorBase<kaldi::BaseFloat>& input,
                 kaldi::VectorBase<kaldi::BaseFloat>* feat) const;
  private:
    DecibelNormalizerOptions opts_;
    size_t dim_;
    std::unique_ptr<FeatureExtractorInterface> base_extractor_;
    kaldi::Vector<kaldi::BaseFloat> waveform_;
};

/*
struct NormalizerOptions {
  std::string mean_std_path;
  NormalizerOptions() :
    mean_std_path("") {}

  void Register(kaldi::OptionsItf* opts) {
    opts->Register("mean-std", &mean_std_path, "mean std file");
  }
};

// todo refactor later (SmileGoat)
class PPNormalizer : public FeatureExtractorInterface {
  public:
    explicit PPNormalizer(const NormalizerOptions& opts, 
                          const std::unique_ptr<FeatureExtractorInterface>& pre_extractor);
    ~PPNormalizer() {}
    virtual void AcceptWavefrom(const kaldi::Vector<kaldi::BaseFloat>& input);
    virtual void Read(kaldi::Vector<kaldi::BaseFloat>* feat);
    virtual size_t Dim() const;
    bool Compute(const kaldi::Vector<kaldi::BaseFloat>& input,
                 kaldi::Vector<kaldi::BaseFloat>>& feat);

  private:
    bool _initialized;
    kaldi::Vector<float> mean_;
    kaldi::Vector<float> variance_;
    NormalizerOptions _opts;
};
*/
}  // namespace ppspeech