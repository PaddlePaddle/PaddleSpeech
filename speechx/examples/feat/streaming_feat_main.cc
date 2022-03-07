// todo refactor, repalce with gtest

#include "frontend/linear_spectrogram.h"
#include "frontend/normalizer.h"
#include "frontend/feature_extractor_interface.h"
#include "kaldi/util/table-types.h"
#include "base/log.h"
#include "base/flags.h"
#include "kaldi/feat/wave-reader.h"
#include "kaldi/util/kaldi-io.h"

DEFINE_string(wav_rspecifier, "", "test wav path");
DEFINE_string(feature_wspecifier, "", "test wav ark");
DEFINE_string(cmvn_path, "./cmvn.ark", "test wav ark");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);
  
  kaldi::SequentialTableReader<kaldi::WaveHolder> wav_reader(FLAGS_wav_rspecifier);
  kaldi::BaseFloatMatrixWriter feat_writer(FLAGS_feature_wspecifier);

  // test feature linear_spectorgram: wave --> decibel_normalizer --> hanning window -->linear_spectrogram --> cmvn
  // --> feature_cache
  int32 num_done = 0, num_err = 0;
  ppspeech::LinearSpectrogramOptions opt;
  opt.frame_opts.frame_length_ms = 20;
  opt.frame_opts.frame_shift_ms = 10;
  ppspeech::DecibelNormalizerOptions db_norm_opt;
  std::unique_ptr<ppspeech::FeatureExtractorInterface> base_feature_extractor(
      new ppspeech::DecibelNormalizer(db_norm_opt));

  std::shared_ptr<ppspeech::FeatureExtractorInterface> linear_spectrogram(
      new ppspeech::LinearSpectrogram(opt, base_feature_extractor));

  std::shared_ptr<ppspeech::FeatureExtractorInterface> cmvn(
      new ppspeech::CMVN(FLAGS_cmvn_path, linear_spectrogram);
  ppspeech::FeatureCache(cmvn);

  float streaming_chunk = 0.36;
  int sample_rate = 16000;
  int chunk_sample_size = streaming_chunk * sample_rate;
  // thread 1 feed feature

  for (; !wav_reader.Done(); wav_reader.Next()) {
    std::string utt = wav_reader.Key();
    const kaldi::WaveData &wave_data = wav_reader.Value();

    if (num_done % 50 == 0 && num_done != 0)
    KALDI_VLOG(2) << "Processed " << num_done << " utterances";
    num_done++;
  }
    KALDI_LOG << "Done " << num_done << " utterances, " << num_err
              << " with errors.";
    return (num_done != 0 ? 0 : 1);
}
