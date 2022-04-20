// Copyright 2019 PEACH LAB. All Rights Reserved.
// Author: goat.zhou@foxmail.com

#include "itn/string_composer.h"

#include <unistd.h>

namespace goat {

using fst::StdArc;
typedef fst::StdVectorFst MutableTransducer;
typedef StdArc::StateId StateId;
typedef StdArc::Weight Weight;

bool FileExists(const string& file_name) {
  return access(file_name.c_str(), 0) == 0;
}

StringComposer::StringComposer(const string& rule_fst_path) {
  if (FileExists(rule_fst_path)) {
    rule_fst_.reset(fst::Fst<StdArc>::Read(rule_fst_path));
    if (!rule_fst_->Properties(fst::kILabelSorted, false)) {
      MutableTransducer* sorted_rule_fst = new MutableTransducer(*rule_fst_);
      ArcSort(sorted_rule_fst, fst::ILabelCompare<StdArc>());
      rule_fst_.reset(static_cast<Transducer*>(sorted_rule_fst));
    }
  } else {
    cout<< "WARNING" << "rule_fst not found.";
  }
}

void StringComposer::ByteLabels2String(const vector<Label>& input_labels,
                                       string* words) const {
  words->reserve(input_labels.size());
  for (auto label : input_labels) {
    words->push_back(label);
  }
}

void StringComposer::String2ByteLabels(const string& input_str,
                                       vector<Label>* labels) const {
  labels->reserve(input_str.size());
  for (char c : input_str) {
    labels->push_back(static_cast<unsigned char> (c));
  }
}


void StringComposer::LongestMatch(const vector<Label>& input_labels,
                                   vector<Label>* output_labels) const {
  std::queue<SearchState> state_queue;
  SearchState init_state(0, rule_fst_->Start());
  SearchState final_state = init_state;
  state_queue.push(init_state);

  while (final_state.offset < input_labels.size()) {
    while (!state_queue.empty()) {
      SearchState cur_state = state_queue.front();
      state_queue.pop();

      if (rule_fst_->Final(cur_state.state_id) == Weight::One()) {
        if (cur_state.offset > final_state.offset) {
          final_state = cur_state;
        }
      }

      Label cur_label = cur_state.offset >= input_labels.size() ?
                              0 : input_labels[cur_state.offset];

      fst::ArcIterator<fst::StdFst> iter(*rule_fst_, cur_state.state_id);
      for (; !iter.Done(); iter.Next()) {
        const StdArc& arc  = iter.Value();
        SearchState next_state(cur_state);
        if (arc.ilabel == eps_label || arc.ilabel == cur_label) {
           if (arc.olabel != eps_label) {
             next_state.output_labels.push_back(arc.olabel);
           }
           if (arc.ilabel == cur_label) {
             next_state.offset += 1;
           }
           next_state.state_id = arc.nextstate;
           state_queue.push(next_state);
        }
      }
    }

    if (final_state.offset == init_state.offset) {
      final_state.offset += 1;
      final_state.output_labels.push_back(input_labels[init_state.offset]);
    }
    init_state = final_state;
    init_state.state_id = rule_fst_->Start();
    state_queue.push(init_state);
  }

  output_labels->reserve(final_state.output_labels.size());
  for (auto x : final_state.output_labels) {
    output_labels->push_back(x);
  }
}

}  // namespace goat
