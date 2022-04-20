// Copyright 2019 PEACH LAB. All Rights Reserved.
// Author: goat.zhou@foxmail.com

#include <fstream>

#include "itn/inverse_text_normalizer_impl.h"

int main(int argv, char** argc) {
  string fst_input = argc[1];
  string text_input = argc[2];
  string result_file = argc[3];

  goat::InverseTextNormalizerImpl test_itn(fst_input);
  std::ifstream fin(text_input);
  if (!fin) {
     cout << "Can not open input file";
     return 1;
  }
  std::ofstream fo(result_file);
  string input_text;
  string result;
  while (getline(fin, input_text)) {
    test_itn.Process(&input_text);
    result = input_text + "\n";
    fo << result;
  }
  cout << result << endl;
  return 0;
}
