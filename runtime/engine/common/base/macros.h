// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <limits>
#include <string>

#ifndef DISALLOW_COPY_AND_ASSIGN
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName&) = delete;    \
    void operator=(const TypeName&) = delete
#endif

namespace ppspeech {

// kSpaceSymbol in UTF-8 is: ▁
const char kSpaceSymbo[] = "\xe2\x96\x81";

}  // namespace ppspeech
