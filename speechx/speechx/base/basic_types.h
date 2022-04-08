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

#include "kaldi/base/kaldi-types.h"

#include <limits>

typedef float BaseFloat;
typedef double double64;

typedef signed char int8;
typedef short int16;
typedef int int32;

#if defined(__LP64__) && !defined(OS_MACOSX) && !defined(OS_OPENBSD)
typedef long int64;
#else
typedef long long int64;
#endif

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;

#if defined(__LP64__) && !defined(OS_MACOSX) && !defined(OS_OPENBSD)
typedef unsigned long uint64;
#else
typedef unsigned long long uint64;
#endif

typedef signed int char32;

const uint8 kuint8max = ((uint8)0xFF);
const uint16 kuint16max = ((uint16)0xFFFF);
const uint32 kuint32max = ((uint32)0xFFFFFFFF);
const uint64 kuint64max = ((uint64)(0xFFFFFFFFFFFFFFFFLL));
const int8 kint8min = ((int8)0x80);
const int8 kint8max = ((int8)0x7F);
const int16 kint16min = ((int16)0x8000);
const int16 kint16max = ((int16)0x7FFF);
const int32 kint32min = ((int32)0x80000000);
const int32 kint32max = ((int32)0x7FFFFFFF);
const int64 kint64min = ((int64)(0x8000000000000000LL));
const int64 kint64max = ((int64)(0x7FFFFFFFFFFFFFFFLL));

const BaseFloat kBaseFloatMax = std::numeric_limits<BaseFloat>::max();
const BaseFloat kBaseFloatMin = std::numeric_limits<BaseFloat>::min();
