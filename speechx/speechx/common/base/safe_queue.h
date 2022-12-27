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

#include "base/common.h"

namespace ppspeech {

template <typename T>
class SafeQueue {
  public:
    explicit SafeQueue(size_t capacity = 0);
    void push_back(const T& in);
    bool pop(T* out);
    bool empty() const { return buffer_.empty(); }
    size_t size() const { return buffer_.size(); }
    void clear();


  private:
    std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<T> buffer_;
    size_t capacity_;
};

template <typename T>
SafeQueue<T>::SafeQueue(size_t capacity) : capacity_(capacity) {}

template <typename T>
void SafeQueue<T>::push_back(const T& in) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (capacity_ > 0 && buffer_.size() == capacity_) {
        condition_.wait(lock, [this] { return capacity_ >= buffer_.size(); });
    }

    buffer_.push_back(in);
    condition_.notify_one();
}

template <typename T>
bool SafeQueue<T>::pop(T* out) {
    if (buffer_.empty()) {
        return false;
    }

    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return buffer_.size() > 0; });
    *out = std::move(buffer_.front());
    buffer_.pop_front();
    condition_.notify_one();
    return true;
}

template <typename T>
void SafeQueue<T>::clear() {
    std::unique_lock<std::mutex> lock(mutex_);
    buffer_.clear();
    condition_.notify_one();
}
}  // namespace ppspeech
