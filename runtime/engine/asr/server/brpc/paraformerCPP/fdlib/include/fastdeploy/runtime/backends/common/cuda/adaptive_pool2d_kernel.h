
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

#include <cuda_fp16.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <vector>

namespace fastdeploy {

void CudaAdaptivePool(const std::vector<int64_t>& input_dims,
                      const std::vector<int64_t>& output_dims, void* output,
                      const void* input, void* compute_stream,
                      const std::string& pooling_type,
                      const std::string& dtype = "float",
                      const std::string& out_dtype = "float");

}  // namespace fastdeploy
