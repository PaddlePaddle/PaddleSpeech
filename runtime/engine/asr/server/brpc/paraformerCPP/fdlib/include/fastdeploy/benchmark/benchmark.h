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
#include "fastdeploy/core/config.h"
#include "fastdeploy/utils/utils.h"
#include "fastdeploy/utils/perf.h"
#include "fastdeploy/benchmark/option.h"
#include "fastdeploy/benchmark/results.h"

#ifdef ENABLE_BENCHMARK
  #define __RUNTIME_PROFILE_LOOP_BEGIN(option, base_loop)               \
    int __p_loop = (base_loop);                                         \
    const bool __p_enable_profile = option.enable_profile;              \
    const bool __p_include_h2d_d2h = option.include_h2d_d2h;            \
    const int __p_repeats = option.repeats;                             \
    const int __p_warmup = option.warmup;                               \
    if (__p_enable_profile && (!__p_include_h2d_d2h)) {                 \
      __p_loop = (__p_repeats) + (__p_warmup);                          \
      FDINFO << option << std::endl;                                    \
    }                                                                   \
    TimeCounter __p_tc;                                                 \
    bool __p_tc_start = false;                                          \
    for (int __p_i = 0; __p_i < __p_loop; ++__p_i) {                    \
      if (__p_i >= (__p_warmup) && (!__p_tc_start)) {                   \
        __p_tc.Start();                                                 \
        __p_tc_start = true;                                            \
      }                                                                 \

  #define __RUNTIME_PROFILE_LOOP_END(result)                            \
    }                                                                   \
    if ((__p_enable_profile && (!__p_include_h2d_d2h))) {               \
      if (__p_tc_start) {                                               \
        __p_tc.End();                                                   \
        double __p_tc_duration = __p_tc.Duration();                     \
        result.time_of_runtime =                                        \
          __p_tc_duration / static_cast<double>(__p_repeats);           \
      }                                                                 \
    }

  #define __RUNTIME_PROFILE_LOOP_H2D_D2H_BEGIN(option, base_loop)       \
    int __p_loop_h = (base_loop);                                       \
    const bool __p_enable_profile_h = option.enable_profile;            \
    const bool __p_include_h2d_d2h_h = option.include_h2d_d2h;          \
    const int __p_repeats_h = option.repeats;                           \
    const int __p_warmup_h = option.warmup;                             \
    if (__p_enable_profile_h && __p_include_h2d_d2h_h) {                \
      __p_loop_h = (__p_repeats_h) + (__p_warmup_h);                    \
      FDINFO << option << std::endl;                                    \
    }                                                                   \
    TimeCounter __p_tc_h;                                               \
    bool __p_tc_start_h = false;                                        \
    for (int __p_i_h = 0; __p_i_h < __p_loop_h; ++__p_i_h) {            \
      if (__p_i_h >= (__p_warmup_h) && (!__p_tc_start_h)) {             \
        __p_tc_h.Start();                                               \
        __p_tc_start_h = true;                                          \
      }                                                                 \

  #define __RUNTIME_PROFILE_LOOP_H2D_D2H_END(result)                    \
    }                                                                   \
    if ((__p_enable_profile_h && __p_include_h2d_d2h_h)) {              \
      if (__p_tc_start_h) {                                             \
         __p_tc_h.End();                                                \
        double __p_tc_duration_h = __p_tc_h.Duration();                 \
        result.time_of_runtime =                                        \
          __p_tc_duration_h / static_cast<double>(__p_repeats_h);       \
      }                                                                 \
    }
#else
  #define __RUNTIME_PROFILE_LOOP_BEGIN(option, base_loop)               \
    for (int __p_i = 0; __p_i < (base_loop); ++__p_i) {
  #define __RUNTIME_PROFILE_LOOP_END(result) }
  #define __RUNTIME_PROFILE_LOOP_H2D_D2H_BEGIN(option, base_loop)       \
    for (int __p_i_h = 0; __p_i_h < (base_loop); ++__p_i_h) {
  #define __RUNTIME_PROFILE_LOOP_H2D_D2H_END(result) }
#endif
