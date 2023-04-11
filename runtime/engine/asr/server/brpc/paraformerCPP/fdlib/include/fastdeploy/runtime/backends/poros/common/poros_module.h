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

#include "torch/csrc/jit/jit_log.h"  // NOLINT
#include "torch/script.h"  // NOLINT
#include <string>
// #include "ATen/Context.h"

namespace baidu {
namespace mirana {
namespace poros {

enum Device : int8_t { GPU = 0, CPU, XPU, UNKNOW };

struct PorosOptions {
  Device device = GPU;
  bool debug = false;
  bool use_fp16 = false;
  bool is_dynamic = false;
  bool long_to_int = true;
  uint64_t max_workspace_size = 1ULL << 30;
  int32_t device_id = -1;
  int32_t unconst_ops_thres = -1;
  bool use_nvidia_tf32 = false;
};


class PorosModule : public torch::jit::Module {
 public:
  PorosModule(torch::jit::Module module) : torch::jit::Module(module) {}  // NOLINT
  ~PorosModule() = default;

  void to_device(Device device) { _options.device = device; }

  // c10::IValue forward(std::vector<c10::IValue> inputs);
  // void save(const std::string& filename);
 public:
  PorosOptions _options;
};

// via porosmodule.save
std::unique_ptr<PorosModule> Load(const std::string& filename,
                                  const PorosOptions& options);

}  // namespace poros
}  // namespace mirana
}  // namespace baidu
