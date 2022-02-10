
#pragma once

#include "frontend/feature_extractor_interface.h"
#include "kaldi/feat/feature-window.h"
#include "base/common.h"

namespace ppspeech {

struct LinearSpectrogramOptions {
    kaldi::FrameExtrationOptions frame_opts;
    LinearSpectrogramOptions():
        frame_opts() {}

    void Register(kaldi::OptionsItf* opts) {
        frame_opts.Register(opts);
    }
};

class LinearSpectrogram : public FeatureExtractorInterface {
  public:
    explict LinearSpectrogram(const LinearSpectrogramOptions& opts,
                              const std::unique_ptr<FeatureExtractorInterface> base_extractor);
    virtual void AcceptWavefrom(const kaldi::Vector<kaldi::BaseFloat>& input);
    virtual void Read(kaldi::Vector<kaldi::BaseFloat>* feat);
    virtual size_t Dim() const { return dim_; }
    void ReadFeats(kaldi::Matrix<kaldi::BaesFloat>* feats) const;

  private: 
    void Hanning(std::vector<kaldi::BaseFloat>& data) const;
    kaldi::int32 Compute(const std::vector<kaldi::BaseFloat>& wave,
                         std::vector<std::vector<kaldi::BaseFloat>>& feat);
    void Compute(const kaldi::VectorBase<kaldi::BaseFloat>& input,
                         kaldi::VectorBae<kaldi::BaseFloat>* feature);
    bool NumpyFft(std::vector<kaldi::BaseFloat>* v,
                  std::vector<kaldi::BaseFloat>* real,
                  std::vector<kaldi::BaseFloat>* img) const;

    kaldi::int32 fft_points_;
    size_t dim_;
    std::vector<kaldi::BaseFloat> hanning_window_;
    kaldi::BaseFloat hanning_window_energy_;
    LinearSpectrogramOptions opts_;
    kaldi::Vector<kaldi::BaseFloat> wavefrom_; // remove later, todo(SmileGoat)
    std::unique_ptr<FeatureExtractorInterface> base_extractor_;
    DISALLOW_COPY_AND_ASSIGN(LinearSpectrogram);
};


}  // namespace ppspeech