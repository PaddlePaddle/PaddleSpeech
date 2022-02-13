
#pragma once

#include "frontend/feature_extractor_interface.h"
#include "kaldi/feat/feature-window.h"
#include "base/common.h"

namespace ppspeech {

struct LinearSpectrogramOptions {
    kaldi::FrameExtractionOptions frame_opts;
    LinearSpectrogramOptions():
        frame_opts() {}

    void Register(kaldi::OptionsItf* opts) {
        frame_opts.Register(opts);
    }
};

class LinearSpectrogram : public FeatureExtractorInterface {
  public:
    explicit LinearSpectrogram(const LinearSpectrogramOptions& opts,
                               std::unique_ptr<FeatureExtractorInterface> base_extractor);
    virtual void AcceptWavefrom(const kaldi::VectorBase<kaldi::BaseFloat>& input);
    virtual void Read(kaldi::VectorBase<kaldi::BaseFloat>* feat);
    virtual size_t Dim() const { return dim_; }
    void ReadFeats(kaldi::Matrix<kaldi::BaseFloat>* feats);

  private: 
    void Hanning(std::vector<kaldi::BaseFloat>* data) const;
    bool Compute(const std::vector<kaldi::BaseFloat>& wave,
                 std::vector<std::vector<kaldi::BaseFloat>>& feat);
    void Compute(const kaldi::Vector<kaldi::BaseFloat>& input,
                 kaldi::Vector<kaldi::BaseFloat>* feature);
    bool NumpyFft(std::vector<kaldi::BaseFloat>* v,
                  std::vector<kaldi::BaseFloat>* real,
                  std::vector<kaldi::BaseFloat>* img) const;

    kaldi::int32 fft_points_;
    size_t dim_;
    std::vector<kaldi::BaseFloat> hanning_window_;
    kaldi::BaseFloat hanning_window_energy_;
    LinearSpectrogramOptions opts_;
    kaldi::Vector<kaldi::BaseFloat> waveform_; // remove later, todo(SmileGoat)
    std::unique_ptr<FeatureExtractorInterface> base_extractor_;
    DISALLOW_COPY_AND_ASSIGN(LinearSpectrogram);
};


}  // namespace ppspeech