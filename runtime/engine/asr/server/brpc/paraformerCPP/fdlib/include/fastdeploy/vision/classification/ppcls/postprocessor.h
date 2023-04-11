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
#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/vision/common/result.h"

namespace fastdeploy {
namespace vision {

namespace classification {
/*! @brief Postprocessor object for PaddleClas serials model.
 */
class FASTDEPLOY_DECL PaddleClasPostprocessor {
 public:
  /** \brief Create a postprocessor instance for PaddleClas serials model
   *
   * \param[in] topk The topk result filtered by the classify confidence score, default 1
   */
  explicit PaddleClasPostprocessor(int topk = 1);

  /** \brief Process the result of runtime and fill to ClassifyResult structure
   *
   * \param[in] tensors The inference result from runtime
   * \param[in] result The output result of classification
   * \return true if the postprocess successed, otherwise false
   */
  bool Run(const std::vector<FDTensor>& tensors,
          std::vector<ClassifyResult>* result);

  /// Set topk value
  void SetTopk(int topk) { topk_ = topk; }

  /// Get topk value
  int GetTopk() const { return topk_; }

 private:
  int topk_ = 1;
  bool initialized_ = false;
};

}  // namespace classification
}  // namespace vision
}  // namespace fastdeploy
