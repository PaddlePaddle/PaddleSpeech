#include "base/basic_types.h"

struct DecoderResult {
  BaseFloat acoustic_score; 
  std::vector<int32> words_idx;
  std::vector<pair<int32, int32>> time_stamp;
};
