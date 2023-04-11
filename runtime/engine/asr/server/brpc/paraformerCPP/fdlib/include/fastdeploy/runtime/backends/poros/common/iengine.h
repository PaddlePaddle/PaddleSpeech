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

#include <string>

// from pytorch
#include "ATen/core/interned_strings.h"  // NOLINT
#include "torch/csrc/jit/ir/ir.h"  // NOLINT
#include "torch/script.h"  // NOLINT

#include "plugin_create.h"  // NOLINT

namespace baidu {
namespace mirana {
namespace poros {

struct PorosGraph {
  torch::jit::Graph* graph = NULL;
  torch::jit::Node* node = NULL;
};

typedef uint64_t EngineID;

class IEngine : public IPlugin, public torch::CustomClassHolder {
 public:
  virtual ~IEngine() {}

  /**
     * @brief init, initialization must be successful if the init is successful
     * @return int
     * @retval 0 => success, <0 => fail
     **/
  virtual int init() = 0;

  /**
     * @brief During compilation, the subgraph is converted into the graph structure of the corresponding engine and stored inside the engine, so that the execute_engine at runtime can be called
     * @param [in] sub_graph  : subgraph
     * @return [res]int
     * @retval 0 => success, <0 => fail
     **/
  virtual int transform(const PorosGraph& sub_graph) = 0;

  /**
     * @brief Subgraph execution period logic
     * @param [in] inputs  : input tensor
     * @return [res] output tensor
     **/
  virtual std::vector<at::Tensor>
  excute_engine(const std::vector<at::Tensor>& inputs) = 0;

  virtual void register_module_attribute(const std::string& name,
                                         torch::jit::Module& module) = 0;

  // Logo
  virtual const std::string who_am_i() = 0;

  // Whether the node is supported by the current engine
  bool is_node_supported(const torch::jit::Node* node);

 public:
  std::pair<uint64_t, uint64_t> _num_io;  // Number of input/output parameters
  EngineID _id;
};

}  // namespace poros
}  // namespace mirana
}  // namespace baidu
