#include "itn/inverse_text_normalizer_impl.h"
#include "itn/blank_processor.h"

namespace goat {

typedef fst::StdArc::Label Label;

InverseTextNormalizerImpl::InverseTextNormalizerImpl(
    const string& rule_fst) {
  normalizer_.reset(new StringComposer(rule_fst));
  blank_processor_.reset(new BlankProcessor());
}

bool InverseTextNormalizerImpl::Process(string* result) {
  string query = *result;
  string itn_query = "";
  blank_processor_->DelBlankInQuery(&query);
  blank_processor_->AddBlankInQuery(&query);

	cout << "del blank: " << query << endl;

  vector<Label> input_labels, output_labels;
  normalizer_->String2ByteLabels(query, &input_labels);
  normalizer_->LongestMatch(input_labels, &output_labels);
  normalizer_->ByteLabels2String(output_labels, &itn_query);

	cout << "after itn : " << itn_query << endl;

  if (*(itn_query.begin()) == ' ') {
    itn_query.erase(itn_query.begin());
  }

  if (*(itn_query.end() - 1) == ' ') {
    itn_query.erase(itn_query.end() - 1);
  }

  blank_processor_->DelBlankInQuery(&itn_query);
  *result = itn_query;
  return true;
}

}
