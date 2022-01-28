
#include "frontend/normalizer.h"

DecibelNormalizer::DecibelNormalizer(
    const DecibelNormalizerOptions& opts,
    const std::unique_ptr<FeatureExtractorInterface>& pre_extractor) {

}
                                    
void DecibelNormalizer::AcceptWavefrom(const kaldi::Vector<kaldi::BaseFloat>& input) {

}

void DecibelNormalizer::Read(kaldi::Vector<kaldi::BaseFloat>* feat) {

}

bool DecibelNormalizer::Compute(const Vector<kaldi::BaseFloat>& input,
                                kaldi::Vector<kaldi::BaseFloat>* feat) {
  // calculate db rms
  float rms_db = 0.0;
  float mean_square = 0.0;
  float gain = 0.0;
  vector<BaseFloat> smaples;
  samples.resize(input.Size());
  for (int32 i = 0; i < samples.size(); ++i) {
    samples[i] = input(i);
  }
  
  // square
  for (auto &d : samples) {
    if (_opts.convert_int_float) {
    d = d * WAVE_FLOAT_NORMALIZATION;
    }
    mean_square += d * d;
  }

  // mean
  mean_square /= samples.size();
  rms_db = 10 * std::log10(mean_square);
  gain = opts.target_db - rms_db;

  if (gain > opts.max_gain_db) {
    LOG(ERROR) << "Unable to normalize segment to " << opts.target_db << "dB,"
                << "because the the probable gain have exceeds opts.max_gain_db" 
                <<  opts.max_gain_db << "dB.";
    return false;
  }

  // Note that this is an in-place transformation.
  for (auto &item : samples) {
    // python item *= 10.0 ** (gain / 20.0)
    item *= std::pow(10.0, gain / 20.0);
  }

  return true;
}


PPNormalizer::PPNormalizer(
    const PPNormalizerOptions& opts,
    const std::unique_ptr<FeatureExtractorInterface>& pre_extractor) {

}
                                    
void PPNormalizer::AcceptWavefrom(const kaldi::Vector<kaldi::BaseFloat>& input) {

}

void PPNormalizer::Read(kaldi::Vector<kaldi::BaseFloat>* feat) {

}

bool PPNormalizer::Compute(const Vector<kaldi::BaseFloat>& input,
                           kaldi::Vector<kaldi::BaseFloat>>* feat) {
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
}
