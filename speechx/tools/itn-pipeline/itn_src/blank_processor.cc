// Copyright 2019 PEACH LAB. All Rights Reserved.
// Author: goat.zhou@foxmail.com

#include "itn/blank_processor.h"

namespace goat {

BlankProcessor::BlankProcessor() {
  char ch = -128;
  legal_english_char_.resize(256);
  legal_arabic_number_.resize(256);
  legal_white_space_.resize(256);
  for (int counter = 0; counter < 256; ++counter, ch++) {
    if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')) {
        legal_english_char_[static_cast<unsigned char>(ch)] = true;
    } else {
        legal_english_char_[static_cast<unsigned char>(ch)] = false;
    }

    if (ch >= '0' && ch <= '9') {
      legal_arabic_number_[static_cast<unsigned char>(ch)] = true;
    } else {
      legal_arabic_number_[static_cast<unsigned char>(ch)] = false;
    }

    if (ch == ' ' || ch == '\t' || ch == '\n' ||
        ch == '\r' || ch == '\v' || ch == '\f') {
       legal_white_space_[static_cast<unsigned char>(ch)] = true;
    } else {
       legal_white_space_[static_cast<unsigned char>(ch)] = false;
    }
  }
}

bool BlankProcessor::IsLegalEnglishChar(char ch) {
  return legal_english_char_[static_cast<unsigned char>(ch)];
}

bool BlankProcessor::IsLegalWhiteSpace(char ch) {
  return legal_white_space_[static_cast<unsigned char>(ch)];
}

bool BlankProcessor::IsLegalArabicNumber(char ch) {
  return legal_arabic_number_[static_cast<unsigned char>(ch)];
}

void BlankProcessor::AddBlankInQuery(string* query) {
  std::ostringstream result_oss;
  result_oss << ' ';
  for (size_t idx = 0; idx < query->size(); ++idx) {
    char ch = (*query)[idx];
    if (!IsLegalWhiteSpace(ch)) {
      result_oss << ch;
  } else {
     result_oss << ch;
     result_oss << ' ';
    }
  }
  result_oss << ' ';
  *query = result_oss.str();
}

void BlankProcessor::DelBlankInQuery(string* query) {
  std::ostringstream result_oss;
  for (size_t idx = 0; idx < query->size(); ++idx) {
    char ch = (*query)[idx];
    if (IsLegalWhiteSpace(ch)) {
      if (idx > 0 && (IsLegalEnglishChar((*query)[idx - 1]) ||
        IsLegalArabicNumber((*query)[idx - 1])) && (idx < query->size() - 1)) {
        while ((idx < query->size() - 1) &&
                IsLegalWhiteSpace((*query)[idx + 1]))
          ++idx;
        if (IsLegalEnglishChar((*query)[idx + 1]) ||
            IsLegalArabicNumber((*query)[idx + 1]))
          result_oss << ' ';
      }
    } else {
      result_oss << ch;
    }
  }
  *query = result_oss.str();
}

}  // namespace goat
