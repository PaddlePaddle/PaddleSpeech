
// Copyright (c) 2021 Mobvoi Inc (Zhendong Peng)
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

#include "utils/math.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <utility>

#include "base/common.h"


namespace ppspeech {

// Sum in log scale
float LogSumExp(float x, float y) {
    if (x <= -kBaseFloatMax) return y;
    if (y <= -kBaseFloatMax) return x;
    float max = std::max(x, y);
    return max + std::log(std::exp(x - max) + std::exp(y - max));
}

// greater compare for smallest priority_queue
template <typename T>
struct ValGreaterComp {
    bool operator()(const std::pair<T, int32_t>& lhs,
                    const std::pair<T, int32_t>& rhs) const {
        return lhs.first > rhs.first ||
               (lhs.first == rhs.first && lhs.second < rhs.second);
    }
};

template <typename T>
void TopK(const std::vector<T>& data,
          int32_t k,
          std::vector<T>* values,
          std::vector<int>* indices) {
    int n = data.size();
    int min_k_n = std::min(k, n);

    // smallest heap, (val, idx)
    std::vector<std::pair<T, int32_t>> smallest_heap;
    for (int i = 0; i < min_k_n; i++) {
        smallest_heap.emplace_back(data[i], i);
    }

    // smallest priority_queue
    std::priority_queue<std::pair<T, int32_t>,
                        std::vector<std::pair<T, int32_t>>,
                        ValGreaterComp<T>>
        pq(ValGreaterComp<T>(), std::move(smallest_heap));

    // top k
    for (int i = k; i < n; i++) {
        if (pq.top().first < data[i]) {
            pq.pop();
            pq.emplace(data[i], i);
        }
    }

    values->resize(min_k_n);
    indices->resize(min_k_n);

    // from largest to samllest
    int cur = values->size() - 1;
    while (!pq.empty()) {
        const auto& item = pq.top();
        pq.pop();

        (*values)[cur] = item.first;
        (*indices)[cur] = item.second;

        cur--;
    }
}

template void TopK<float>(const std::vector<float>& data,
                          int32_t k,
                          std::vector<float>* values,
                          std::vector<int>* indices);

}  // namespace ppspeech