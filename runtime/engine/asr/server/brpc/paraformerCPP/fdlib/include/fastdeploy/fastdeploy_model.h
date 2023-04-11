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
#include "fastdeploy/runtime.h"

namespace fastdeploy {

/*! @brief Base model object for all the vision models
 */
class FASTDEPLOY_DECL FastDeployModel {
 public:
  /// Get model's name
  virtual std::string ModelName() const { return "NameUndefined"; }

  /** \brief Inference the model by the runtime. This interface is included in the `Predict()` function, so we don't call `Infer()` directly in most common situation
  */
  virtual bool Infer(std::vector<FDTensor>& input_tensors,
                     std::vector<FDTensor>* output_tensors);

  /** \brief Inference the model by the runtime. This interface is using class member reused_input_tensors_ to do inference and writing results to reused_output_tensors_
  */
  virtual bool Infer();

  RuntimeOption runtime_option;
  /** \brief Model's valid cpu backends. This member defined all the cpu backends have successfully tested for the model
   */
  std::vector<Backend> valid_cpu_backends = {Backend::ORT};
  /** Model's valid gpu backends. This member defined all the gpu backends have successfully tested for the model
   */
  std::vector<Backend> valid_gpu_backends = {Backend::ORT};
  /** Model's valid ipu backends. This member defined all the ipu backends have successfully tested for the model
   */
  std::vector<Backend> valid_ipu_backends = {};
  /** Model's valid timvx backends. This member defined all the timvx backends have successfully tested for the model
   */
  std::vector<Backend> valid_timvx_backends = {};
    /** Model's valid directml backends. This member defined all the onnxruntime directml backends have successfully tested for the model
   */
  std::vector<Backend> valid_directml_backends = {};
  /** Model's valid ascend backends. This member defined all the cann backends have successfully tested for the model
   */
  std::vector<Backend> valid_ascend_backends = {};
  /** Model's valid KunlunXin xpu backends. This member defined all the KunlunXin xpu backends have successfully tested for the model
   */
  std::vector<Backend> valid_kunlunxin_backends = {};
  /** Model's valid hardware backends. This member defined all the gpu backends have successfully tested for the model
   */
  std::vector<Backend> valid_rknpu_backends = {};
  /** Model's valid hardware backends. This member defined all the sophgo npu backends have successfully tested for the model
   */
  std::vector<Backend> valid_sophgonpu_backends = {};

  /// Get number of inputs for this model
  virtual int NumInputsOfRuntime() { return runtime_->NumInputs(); }
  /// Get number of outputs for this model
  virtual int NumOutputsOfRuntime() { return runtime_->NumOutputs(); }
  /// Get input information for this model
  virtual TensorInfo InputInfoOfRuntime(int index) {
    return runtime_->GetInputInfo(index);
  }
  /// Get output information for this model
  virtual TensorInfo OutputInfoOfRuntime(int index) {
    return runtime_->GetOutputInfo(index);
  }
  /// Check if the model is initialized successfully
  virtual bool Initialized() const {
    return runtime_initialized_ && initialized;
  }

  /** \brief This is a debug interface, used to record the time of runtime (backend + h2d + d2h)
   *
   * example code @code
   * auto model = fastdeploy::vision::PPYOLOE("model.pdmodel", "model.pdiparams", "infer_cfg.yml");
   * if (!model.Initialized()) {
   *   std::cerr << "Failed to initialize." << std::endl;
   *   return -1;
   * }
   * model.EnableRecordTimeOfRuntime();
   * cv::Mat im = cv::imread("test.jpg");
   * for (auto i = 0; i < 1000; ++i) {
   *   fastdeploy::vision::DetectionResult result;
   *   model.Predict(&im, &result);
   * }
   * model.PrintStatisInfoOfRuntime();
   * @endcode After called the `PrintStatisInfoOfRuntime()`, the statistical information of runtime will be printed in the console
   */
  virtual void EnableRecordTimeOfRuntime() {
    time_of_runtime_.clear();
    std::vector<double>().swap(time_of_runtime_);
    enable_record_time_of_runtime_ = true;
  }

  /** \brief Disable to record the time of runtime, see `EnableRecordTimeOfRuntime()` for more detail
  */
  virtual void DisableRecordTimeOfRuntime() {
    enable_record_time_of_runtime_ = false;
  }

  /** \brief Print the statistic information of runtime in the console, see function `EnableRecordTimeOfRuntime()` for more detail
  */
  virtual std::map<std::string, float> PrintStatisInfoOfRuntime();

  /** \brief Check if the `EnableRecordTimeOfRuntime()` method is enabled.
  */
  virtual bool EnabledRecordTimeOfRuntime() {
    return enable_record_time_of_runtime_;
  }
  /** \brief Get profile time of Runtime after the profile process is done.
   */
  virtual double GetProfileTime() {
    return runtime_->GetProfileTime();
  }
  /** \brief Release reused input/output buffers
  */
  virtual void ReleaseReusedBuffer() {
    std::vector<FDTensor>().swap(reused_input_tensors_);
    std::vector<FDTensor>().swap(reused_output_tensors_);
  }

  virtual fastdeploy::Runtime* CloneRuntime() { return runtime_->Clone(); }

  virtual bool SetRuntime(fastdeploy::Runtime* clone_runtime) {
    runtime_ = std::unique_ptr<Runtime>(clone_runtime);
    return true;
  }

  virtual std::unique_ptr<FastDeployModel> Clone() {
    FDERROR << ModelName() << " doesn't support Cone() now." << std::endl;
    return nullptr;
  }

 protected:
  virtual bool InitRuntime();

  bool initialized = false;
  // Reused input tensors
  std::vector<FDTensor> reused_input_tensors_;
  // Reused output tensors
  std::vector<FDTensor> reused_output_tensors_;

 private:
  bool InitRuntimeWithSpecifiedBackend();
  bool InitRuntimeWithSpecifiedDevice();
  bool CreateCpuBackend();
  bool CreateGpuBackend();
  bool CreateIpuBackend();
  bool CreateRKNPUBackend();
  bool CreateSophgoNPUBackend();
  bool CreateTimVXBackend();
  bool CreateKunlunXinBackend();
  bool CreateASCENDBackend();
  bool CreateDirectMLBackend();
  bool IsSupported(const std::vector<Backend>& backends,
                   Backend backend);

  std::shared_ptr<Runtime> runtime_;
  bool runtime_initialized_ = false;
  // whether to record inference time
  bool enable_record_time_of_runtime_ = false;
  std::vector<double> time_of_runtime_;
};

}  // namespace fastdeploy
