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
/*! @brief Postprocessor object for YOLOv5Cls serials model.
 */
class FASTDEPLOY_DECL YOLOv5ClsPostprocessor {
 public:
  /** \brief Create a postprocessor instance for YOLOv5Cls serials model
   */
  YOLOv5ClsPostprocessor();

  /** \brief Process the result of runtime and fill to ClassifyResult structure
   *
   * \param[in] tensors The inference result from runtime
   * \param[in] result The output result of classification
   * \param[in] ims_info The shape info list, record input_shape and output_shape
   * \return true if the postprocess successed, otherwise false
   */
  bool Run(const std::vector<FDTensor>& tensors,
      std::vector<ClassifyResult>* results,
      const std::vector<std::map<std::string, std::array<float, 2>>>& ims_info);

  /// Set topk, default 1
  void SetTopK(const int& topk) {
    topk_ = topk;
  }

  /// Get topk, default 1
  float GetTopK() const { return topk_; }

 protected:
  int topk_;
};

}  // namespace classification
}  // namespace vision
}  // namespace fastdeploy
