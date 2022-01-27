
#include "frontend/linear_spectrogram.h"
#include "kaldi/util/table-types.h"
#include "base/log.h"
#include "base/flags.h"
#include "kaldi/feat/wave-reader.h"

DEFINE_string(wav_rspecifier, "", "test wav path");
DEFINE_string(feature_wspecifier, "", "test wav ark");

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);
  
  kaldi::SequentialTableReader<kaldi::WaveHolder> wav_reader(FLAGS_wav_rspecifier);
  kaldi::BaseFloatMatrixWriter feat_writer(FLAGS_feature_wspecifier);

  int32 num_done = 0, num_err = 0;
  ppspeech::LinearSpectrogramOptions opt;
  ppspeech::LinearSpectrogram linear_spectrogram(opt);
  for (; !wav_reader.Done(); wav_reader.Next()) {
    std::string utt = wav_reader.Key();
    const kaldi::WaveData &wave_data = wav_reader.Value();

    int32 this_channel = 0;
    kaldi::SubVector<kaldi::BaseFloat> waveform(wave_data.Data(), this_channel);
    kaldi::Matrix<BaseFloat> features;
    linear_spectrogram.AcceptWaveform(waveform);
    linear_spectrogram.ReadFeats(&features);

    feat_writer.Write(utt, features);
    if (num_done % 50 == 0 && num_done != 0)
    KALDI_VLOG(2) << "Processed " << num_done << " utterances";
    num_done++;
  }
    KALDI_LOG << "Done " << num_done << " utterances, " << num_err
              << " with errors.";
    return (num_done != 0 ? 0 : 1);
}