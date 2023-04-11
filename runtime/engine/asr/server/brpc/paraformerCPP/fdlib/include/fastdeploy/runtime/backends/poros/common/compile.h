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

#include <algorithm>
#include <set>
#include <string>
#include <unordered_map>

#include "iengine.h"  // NOLINT
#include "poros_module.h"  // NOLINT
#include "torch/script.h"  // NOLINT

namespace baidu {
namespace mirana {
namespace poros {

/**
 * @brief  compile graph
 *
 * @param [in] module : original module
 * @param [in] input_ivalues : prewarm datas
 * @param [in] options : Inference options
 * @return porosmodule
 * @retval !nullptr => succeed  nullptr => failed
 **/
std::unique_ptr<PorosModule>
Compile(const torch::jit::Module& module,
        const std::vector<std::vector<c10::IValue>>& prewarm_datas,
        const PorosOptions& options);

class Compiler {
 public:
  typedef std::unordered_map<const torch::jit::Node*, IEngine*> engine_map_t;
  typedef std::vector<std::vector<c10::IValue>> ivalue_vec_t;

  Compiler() : _origin_module(NULL) {}
  ~Compiler();

  /**
     * @brief initial Compiler
     *
     * @param [in] options : poros options
     * @return  int
     * @retval 0 => succeed  <0 => failed
    **/
  int init(const PorosOptions& options);

  /**
     * @brief compile whole graph
     *
     * @param [in] origin_module
     * @param [in] prewarm_datas : ivalue_vec_t, vector of IValue
     * @param [out] optimized_module : optimized graph
     * @return  int
     * @retval 0 => succeed  <0 => failed
    **/
  int compile(const torch::jit::Module& origin_module,
              const ivalue_vec_t& prewarm_datas,
              torch::jit::Module* optimized_module);

 private:
  /**
     * @brief preprocess this calculation graph
     *
     * @param [in] prewarm_datas : ivalue_vec_t, vector of IValue
     * @param [out] graph : preprcessed graph
     * @return  int
     * @retval 0 => succeed  <0 => failed
    **/
  int preprocess_graph(const ivalue_vec_t& prewarm_datas,
                       std::shared_ptr<torch::jit::Graph>& graph);

  /**
     * @brief segement this calculation graph
     *
     * @param [in/out] graph
     * @return  int
     * @retval 0 => succeed  <0 => failed
    **/
  int segment_graph(std::shared_ptr<torch::jit::Graph>& graph);

  // Split subgraph（block)
  // The divided subgraph, as a subgraph, is associated with the block
  int segment_block(torch::jit::Block& block, IEngine* engine,
                    int current_depth);

  // Subgraph optimization
  /**
     * @brief Subgraph optimization
     *
     * @param [in] prewarm_datas : ivalue_vec_t, vector of IValue
     * @param [in] opt_graph : ivalue_vec_t, vector of IValue
     * @param [out] optimized_module : optimized graph
     * @return  int
     * @retval 0 => succeed  <0 => failed
    **/
  int optimize_subgraph(const ivalue_vec_t& prewarm_datas,
                        const std::shared_ptr<torch::jit::Graph>& opt_graph,
                        torch::jit::Module* optimized_module);

  // Subgraph optimization(block)
  int optimize_subblock(torch::jit::Block* block,
                        torch::jit::Module* optimized_module);

  /**
     * @brief Compile the subgraph into a new graph based on the engine
     *
     * @param [in] engine : The engine used by the subgraph
     * @param [in] subgraph_node : Subgraph node
     * @return [out] module : Transformed model
     * @retval 0 => succeed  <0 => failed
    **/
  int transform(IEngine* engine, torch::jit::Node& subgraph_node,
                torch::jit::Module& module);

  /**
     * @brief Select engine based on subgraph and options
     *
     * @param [in] node : Jit Node
     * @return  int
     * @retval 0 => succeed  <0 => failed
    **/
  IEngine* select_engine(const torch::jit::Node* n);

  /**
     * @brief destory
     *
     * @return  void
    **/
  void close();

 private:
  int _max_segment_depth{5};    // Maximum subgraph segmentation depth
  ivalue_vec_t _prewarm_datas;  // Prewarm datas
  PorosOptions _options;
  engine_map_t _engine_map;  // The engine used to record the subgraph
  const torch::jit::Module* _origin_module;  // Origin_module
  std::atomic<int> _engine_index = {0};      // Record engine index
};

/**
 * @brief  compile graph, internal use
 *
 * @param [in] module : Origin module
 * @param [in] input_ivalues : Prewarm datas
 * @param [in] options : Inference options
 * @return optimized_module
 * @retval !nullptr => succeed  nullptr => failed
 **/
std::unique_ptr<torch::jit::Module>
CompileGraph(const torch::jit::Module& module,
             const std::vector<std::vector<c10::IValue>>& prewarm_datas,
             const PorosOptions& options);

}  // namespace poros
}  // namespace mirana
}  // namespace baidu
