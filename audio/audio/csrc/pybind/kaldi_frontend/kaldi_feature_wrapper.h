
#include "base/kaldi-common.h"
#include "kaldi_frontend/feature_common.h"
#include "feat/feature-fbank.h"

#pragma once

namespace paddleaudio {

typedef StreamingFeatureTpl<kaldi::FbankComputer> Fbank;

class KaldiFeatureWrapper {
  public:
    static KaldiFeatureWrapper* GetInstance();
    bool InitFbank(kaldi::FbankOptions opts);
    py::array_t<double> ComputeFbank(const py::array_t<double> wav);
    int Dim() {
      return fbank_->Dim();
    }
    void ResetFbank() {
      fbank_->Reset();
    }

  private:
    std::unique_ptr<paddleaudio::Fbank> fbank_;
};

}  // namespace paddleaudio