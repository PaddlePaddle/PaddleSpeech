
#include "frontend/normalizer.h"

namespace ppspeech {

using kaldi::Vector;
using kaldi::BaseFloat;
using std::vector;

DecibelNormalizer::DecibelNormalizer(const DecibelNormalizerOptions& opts) {
  opts_ = opts;
}
                                    
void DecibelNormalizer::AcceptWavefrom(const Vector<BaseFloat>& input) {
  waveform_ = input;
}

void DecibelNormalizer::Read(Vector<BaseFloat>* feat) {
  if (waveform_.Dim() == 0) return;
  Compute(waveform_, feat);
}

//todo remove later
void CopyVector2StdVector(const kaldi::Vector<BaseFloat>& input,
                          vector<BaseFloat>* output) {
  if (input.Dim() == 0) return;
  output->resize(input.Dim());
  for (size_t idx = 0; idx < input.Dim(); ++idx) {
    (*output)[idx] = input(idx);
  }
}

void CopyStdVector2Vector(const vector<BaseFloat>& input,
                          Vector<BaseFloat>* output) {
  if (input.empty()) return;
  output->Resize(input.size());
  for (size_t idx = 0; idx < input.size(); ++idx) {
    (*output)(idx) = input[idx];
  }
}

bool DecibelNormalizer::Compute(const Vector<BaseFloat>& input,
                                Vector<BaseFloat>* feat) const {
  // calculate db rms
  BaseFloat rms_db = 0.0;
  BaseFloat mean_square = 0.0;
  BaseFloat gain = 0.0;
  BaseFloat wave_float_normlization = 1.0f / (std::pow(2, 16 - 1));

  vector<BaseFloat> samples;
  samples.resize(input.Dim());
  for (int32 i = 0; i < samples.size(); ++i) {
    samples[i] = input(i);
  }
  
  // square
  for (auto &d : samples) {
    if (opts_.convert_int_float) {
    d = d * wave_float_normlization;
    }
    mean_square += d * d;
  }

  // mean
  mean_square /= samples.size();
  rms_db = 10 * std::log10(mean_square);
  gain = opts_.target_db - rms_db;

  if (gain > opts_.max_gain_db) {
    LOG(ERROR) << "Unable to normalize segment to " << opts_.target_db << "dB,"
                << "because the the probable gain have exceeds opts_.max_gain_db" 
                <<  opts_.max_gain_db << "dB.";
    return false;
  }

  // Note that this is an in-place transformation.
  for (auto &item : samples) {
    // python item *= 10.0 ** (gain / 20.0)
    item *= std::pow(10.0, gain / 20.0);
  }
  
  CopyStdVector2Vector(samples, feat);
  return true;
}

/*
PPNormalizer::PPNormalizer(
    const PPNormalizerOptions& opts,
    const std::unique_ptr<FeatureExtractorInterface>& pre_extractor) {

}
                                    
void PPNormalizer::AcceptWavefrom(const Vector<BaseFloat>& input) {

}

void PPNormalizer::Read(Vector<BaseFloat>* feat) {

}

bool PPNormalizer::Compute(const Vector<BaseFloat>& input,
                           Vector<BaseFloat>>* feat) {
   if ((input.Dim() % mean_.Dim()) == 0) {
        LOG(ERROR) << "CMVN dimension is wrong!";
        return false;
   }

    try {
      int32 size = mean_.Dim();
      feat->Resize(input.Dim());
      for (int32 row_idx = 0; row_idx < j; ++row_idx) {
        int32 base_idx  = row_idx * size;
        for (int32 idx = 0; idx < mean_.Dim(); ++idx) {
          (*feat)(base_idx + idx) = (input(base_dix + idx) - mean_(idx))* variance_(idx);
        }       
      }

    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
        return false;
    }

    return true;
}*/

} // namespace ppspeech