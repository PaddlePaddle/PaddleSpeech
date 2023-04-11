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

/** \brief All C++ FastDeploy benchmark profile APIs are defined inside this namespace
*
*/
namespace benchmark {

/*! @brief Option object used to control the behavior of the benchmark profiling.
 */
struct BenchmarkOption {
  int warmup = 50;              ///< Warmup for backend inference.
  int repeats = 100;            ///< Repeats for backend inference.
  bool enable_profile = false;  ///< Whether to use profile or not.
  bool include_h2d_d2h = false;  ///< Whether to include time of H2D_D2H for time of runtime. // NOLINT

  friend std::ostream& operator<<(
    std::ostream& output, const BenchmarkOption &option) {
    if (!option.include_h2d_d2h) {
      output << "Running profiling for Runtime "
             << "without H2D and D2H, ";
    } else {
      output << "Running profiling for Runtime "
             << "with H2D and D2H, ";
    }
    output << "Repeats: " << option.repeats << ", "
           << "Warmup: " << option.warmup;
    return output;
  }
};

}  // namespace benchmark
}  // namespace fastdeploy
