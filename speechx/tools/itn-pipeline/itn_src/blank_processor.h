// Copyright 2019 PEACH LAB. All Rights Reserved.
// Author: goat.zhou@foxmail.com

#ifndef ITN_BLANK_PROCESSOR_H_
#define ITN_BLANK_PROCESSOR_H_ 1

#include "base/compat.h"

namespace goat {

class BlankProcessor {
 public:
  BlankProcessor();
  ~BlankProcessor() {};
  void AddBlankInQuery(string* query);
  void DelBlankInQuery(string* query);

 private:
  bool IsLegalEnglishChar(char ch);
  bool IsLegalWhiteSpace(char ch);
  bool IsLegalArabicNumber(char ch);
  vector<bool> legal_english_char_;
  vector<bool> legal_white_space_;
  vector<bool> legal_arabic_number_;
};

}  // namespace goat 
#endif  // ITN_BLANK_PROCESSOR_H_
