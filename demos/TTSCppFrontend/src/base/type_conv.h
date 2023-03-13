// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef BASE_TYPE_CONVC_H
#define BASE_TYPE_CONVC_H

#include <codecvt>
#include <locale>
#include <string>


namespace ppspeech {
// wstring to string
std::string wstring2utf8string(const std::wstring& str);

// string to wstring
std::wstring utf8string2wstring(const std::string& str);
}

#endif  // BASE_TYPE_CONVC_H