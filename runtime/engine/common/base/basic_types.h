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

#include "kaldi/base/kaldi-types.h"

typedef float BaseFloat;
typedef double double64;

typedef signed char int8;
typedef short int16;  // NOLINT
typedef int int32;    // NOLINT

#if defined(__LP64__) && !defined(OS_MACOSX) && !defined(OS_OPENBSD)
typedef long int64;  // NOLINT
#else
typedef long long int64;  // NOLINT
#endif

typedef unsigned char uint8;    // NOLINT
typedef unsigned short uint16;  // NOLINT
typedef unsigned int uint32;    // NOLINT

#if defined(__LP64__) && !defined(OS_MACOSX) && !defined(OS_OPENBSD)
typedef unsigned long uint64;  // NOLINT
#else
typedef unsigned long long uint64;  // NOLINT
#endif

typedef signed int char32;

const uint8 kuint8max = static_cast<uint8>(0xFF);
const uint16 kuint16max = static_cast<uint16>(0xFFFF);
const uint32 kuint32max = static_cast<uint32>(0xFFFFFFFF);
const uint64 kuint64max = static_cast<uint64>(0xFFFFFFFFFFFFFFFFLL);
const int8 kint8min = static_cast<int8>(0x80);
const int8 kint8max = static_cast<int8>(0x7F);
const int16 kint16min = static_cast<int16>(0x8000);
const int16 kint16max = static_cast<int16>(0x7FFF);
const int32 kint32min = static_cast<int32>(0x80000000);
const int32 kint32max = static_cast<int32>(0x7FFFFFFF);
const int64 kint64min = static_cast<int64>(0x8000000000000000LL);
const int64 kint64max = static_cast<int64>(0x7FFFFFFFFFFFFFFFLL);

const BaseFloat kBaseFloatMax = std::numeric_limits<BaseFloat>::max();
const BaseFloat kBaseFloatMin = std::numeric_limits<BaseFloat>::min();
