// todo refactor, repalce with gtest

#include "decoder/ctc_beam_search_decoder.h"
#include "kaldi/util/table-types.h"
#include "base/log.h"
#include "base/flags.h"

DEFINE_string(feature_respecifier, "", "test nnet prob");

using kaldi::BaseFloat;

void SplitFeature(kaldi::Matrix<BaseFloat> feature, 
                  int32 chunk_size,
                  std::vector<kaldi::Matrix<BaseFloat>> feature_chunks) {

}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);
  
  kaldi::SequentialBaseFloatMatrixReader feature_reader(FLAGS_feature_respecifier);

  // test nnet_output --> decoder result
  int32 num_done = 0, num_err = 0;
  
  CTCBeamSearchOptions opts;
  CTCBeamSearch decoder(opts);
  
  ModelOptions model_opts;
  std::shared_ptr<PaddleNnet> nnet(new PaddleNnet(model_opts));

  Decodable decodable();
  decodable.SetNnet(nnet);

  int32 chunk_size = 0;
  for (; !feature_reader.Done(); feature_reader.Next()) {
    string utt = feature_reader.Key();
    const kaldi::Matrix<BaseFloat> feature = feature_reader.Value();
    vector<Matrix<BaseFloat>> feature_chunks;
    SplitFeature(feature, chunk_size, &feature_chunks);  
    for (auto feature_chunk : feature_chunks) {
      decodable.FeedFeatures(feature_chunk);
      decoder.InitDecoder();
      decoder.AdvanceDecode(decodable, chunk_size);
    }
    decodable.InputFinished();
    std::string result;
    result = decoder.GetFinalBestPath();
    KALDI_LOG << " the result of " << utt << " is " << result;
    decodable.Reset();
    ++num_done;
  }

  KALDI_LOG << "Done " << num_done << " utterances, " << num_err
            << " with errors.";
  return (num_done != 0 ? 0 : 1);
}