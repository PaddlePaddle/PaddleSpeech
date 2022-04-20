#include "fst/fstlib.h"
#include "base/compat.h"

namespace goat {

const fst::StdArc::Label eps_label = 0;

class StringComposer {
  public:
   typedef fst::StdArc::Label Label;
   typedef fst::StdFst Transducer;
   typedef fst::StdArc::StateId StateId;

   struct SearchState {
     SearchState() : offset(0), state_id(0) {}
     SearchState(size_t offset, StateId state_id) :
         offset(offset), state_id(state_id) {}

     SearchState& operator = (const SearchState& state) {
       offset = state.offset;
       state_id = state.state_id;
			 output_labels = state.output_labels;
       return *this;
     }

     size_t offset;
     StateId state_id;
     vector<Label> output_labels;
   };

  explicit StringComposer(const string& rule_fst);
  void String2ByteLabels(const string& input_str, vector<Label>* labels) const;
  void ByteLabels2String(const vector<Label>& input_labels,string* words) const;
  void LongestMatch(const vector<Label>& input_labels,
                    vector<Label>* output_labels) const;
  private:
    unique_ptr<Transducer> rule_fst_;
};
    
} // namespace name
