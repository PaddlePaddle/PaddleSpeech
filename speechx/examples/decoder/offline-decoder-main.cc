// todo refactor, repalce with gtest

#include "decoder/ctc_beam_search_decoder.h"
#include "kaldi/util/table-types.h"
#include "base/log.h"
#include "base/flags.h"
#include "nnet/paddle_nnet.h"
#include "nnet/decodable.h"

DEFINE_string(feature_respecifier, "", "test nnet prob");

using kaldi::BaseFloat;
using kaldi::Matrix;
using std::vector;

//void SplitFeature(kaldi::Matrix<BaseFloat> feature, 
//                  int32 chunk_size,
//                  std::vector<kaldi::Matrix<BaseFloat>* feature_chunks) {

//}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);
  
  kaldi::SequentialBaseFloatMatrixReader feature_reader(FLAGS_feature_respecifier);

  // test nnet_output --> decoder result
  int32 num_done = 0, num_err = 0;
  
  ppspeech::CTCBeamSearchOptions opts;
  ppspeech::CTCBeamSearch decoder(opts);
  
  ppspeech::ModelOptions model_opts;
  std::shared_ptr<ppspeech::PaddleNnet> nnet(new ppspeech::PaddleNnet(model_opts));

  std::shared_ptr<ppspeech::Decodable> decodable(new ppspeech::Decodable(nnet));

  //int32 chunk_size = 35;
  decoder.InitDecoder();
  for (; !feature_reader.Done(); feature_reader.Next()) {
    string utt = feature_reader.Key();
    const kaldi::Matrix<BaseFloat> feature = feature_reader.Value();
    decodable->FeedFeatures(feature);
    decoder.AdvanceDecode(decodable, 8);
    decodable->InputFinished();
    std::string result;
    result = decoder.GetFinalBestPath();
    KALDI_LOG << " the result of " << utt << " is " << result;
    decodable->Reset();
    ++num_done;
  }

  KALDI_LOG << "Done " << num_done << " utterances, " << num_err
            << " with errors.";
  return (num_done != 0 ? 0 : 1);
}