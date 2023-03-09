// Copyright      2020  Xiaomi Corporation (authors: Haowen Qiu)
//                      Mobvoi Inc.        (authors: Fangjun Kuang)
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


#include <chrono>

#include "common/utils/timer.h"

namespace ppspeech{

struct TimerImpl{
    TimerImpl() = default;
    virtual ~TimerImpl() = default;
    virtual void Reset() = 0;
    // time in seconds
    virtual double Elapsed() = 0;
};

class CpuTimerImpl : public TimerImpl {
 public:
  CpuTimerImpl() { Reset(); }

  using high_resolution_clock = std::chrono::high_resolution_clock;

  void Reset() override { begin_ = high_resolution_clock::now(); }

  // time in seconds
  double Elapsed() override {
    auto end = high_resolution_clock::now();
    auto dur =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin_);
    return dur.count() / 1000000.0;
  }

 private:
  high_resolution_clock::time_point begin_;
};

Timer::Timer() {
    impl_ = std::make_unique<CpuTimerImpl>();
}

Timer::~Timer() = default;

void Timer::Reset() const { impl_->Reset(); }

double Timer::Elapsed() const { return impl_->Elapsed(); }


} //namespace ppspeech