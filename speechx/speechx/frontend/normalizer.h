
#pragma once

#include "frontend/feature_extractor_interface.h"

namespace ppspeech {


struct DecibelNormalizerOptions {
  float target_db;
  float max_gain_db;
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
    explict DecibelNormalizer(const DecibelNormalizerOptions& opts,
                              const std::unique_ptr<FeatureExtractorInterface>& pre_extractor);
    virtual void AcceptWavefrom(const kaldi::Vector<kaldi::BaseFloat>& input);
    virtual void Read(kaldi::Vector<kaldi::BaseFloat>* feat);
    virtual size_t Dim() const;
    bool Compute(const kaldi::Vector<kaldi::BaseFloat>& input,
                 kaldi::Vector<kaldi::BaseFloat>>* feat);
  private:
};

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

}  // namespace ppspeech