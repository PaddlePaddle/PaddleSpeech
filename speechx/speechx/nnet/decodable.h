#include "nnet/decodable-itf.h"

#include "base/common.h"

namespace ppsepeech {
  struct DecodeableConfig;

  class Decodeable : public kaldi::DecodableInterface {
    public: 
      virtual Init(Decodeable config) = 0;
      virtual Acceptlikeihood() = 0;
    private:
      std::share_ptr<FeatureExtractorInterface> frontend_;
      std::share_ptr<NnetInterface> nnet_;
      //Cache nnet_cache_;
  }

}  // namespace ppspeech