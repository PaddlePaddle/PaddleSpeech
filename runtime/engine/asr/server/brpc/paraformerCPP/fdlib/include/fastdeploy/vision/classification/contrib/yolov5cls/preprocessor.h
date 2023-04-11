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
/*! @brief Preprocessor object for YOLOv5Cls serials model.
 */
class FASTDEPLOY_DECL YOLOv5ClsPreprocessor {
 public:
  /** \brief Create a preprocessor instance for YOLOv5Cls serials model
   */
  YOLOv5ClsPreprocessor();

  /** \brief Process the input image and prepare input tensors for runtime
   *
   * \param[in] images The input image data list, all the elements are returned by cv::imread()
   * \param[in] outputs The output tensors which will feed in runtime
   * \param[in] ims_info The shape info list, record input_shape and output_shape
   * \return true if the preprocess successed, otherwise false
   */
  bool Run(std::vector<FDMat>* images, std::vector<FDTensor>* outputs,
           std::vector<std::map<std::string, std::array<float, 2>>>* ims_info);

  /// Set target size, tuple of (width, height), default size = {224, 224}
  void SetSize(const std::vector<int>& size) { size_ = size; }

  /// Get target size, tuple of (width, height), default size = {224, 224}
  std::vector<int> GetSize() const { return size_; }

 protected:
  bool Preprocess(FDMat* mat, FDTensor* output,
                  std::map<std::string, std::array<float, 2>>* im_info);

  // target size, tuple of (width, height), default size = {224, 224}
  std::vector<int> size_;
};

}  // namespace classification
}  // namespace vision
}  // namespace fastdeploy
