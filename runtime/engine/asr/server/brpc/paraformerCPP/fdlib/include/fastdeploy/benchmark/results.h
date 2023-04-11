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

namespace fastdeploy {
namespace benchmark {

/*! @brief Result object used to record the time of runtime after benchmark profiling is done.
 */
struct BenchmarkResult {
  ///< Means pure_backend_time+time_of_h2d_d2h(if include_h2d_d2h=true).
  double time_of_runtime = 0.0f; 
};

} // namespace benchmark
} // namespace fastdeploy