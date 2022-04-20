#ifndef ITN_INVERSE_TEXT_NORMALIZER_IMPL_H_
#define ITN_INVERSE_TEXT_NORMALIZER_IMPL_H_ 1

#include "itn/blank_processor.h"
#include "itn/string_composer.h"

namespace goat {

class InverseTextNormalizerImpl {
 public:
  InverseTextNormalizerImpl(const string& rule_fst);
	~InverseTextNormalizerImpl() {}
	bool Process(string* result);
 private:
  unique_ptr<StringComposer> normalizer_;
  unique_ptr<BlankProcessor> blank_processor_;
};

}  // namespace goat
#endif  // ITN_INVERSE_TEXT_NORMALIZER_IMPL_H_
