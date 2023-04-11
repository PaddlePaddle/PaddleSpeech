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

/*! \file runtime_option.h
    \brief A brief file description.

    More details
 */

#pragma once

#include <algorithm>
#include <map>
#include <vector>
#include "fastdeploy/runtime/enum_variables.h"
#include "fastdeploy/runtime/backends/lite/option.h"
#include "fastdeploy/runtime/backends/openvino/option.h"
#include "fastdeploy/runtime/backends/ort/option.h"
#include "fastdeploy/runtime/backends/paddle/option.h"
#include "fastdeploy/runtime/backends/poros/option.h"
#include "fastdeploy/runtime/backends/rknpu2/option.h"
#include "fastdeploy/runtime/backends/sophgo/option.h"
#include "fastdeploy/runtime/backends/tensorrt/option.h"
#include "fastdeploy/benchmark/option.h"

namespace fastdeploy {

/*! @brief Option object used when create a new Runtime object
 */
struct FASTDEPLOY_DECL RuntimeOption {
  /** \brief Set path of model file and parameter file
   *
   * \param[in] model_path Path of model file, e.g ResNet50/model.pdmodel for Paddle format model / ResNet50/model.onnx for ONNX format model
   * \param[in] params_path Path of parameter file, this only used when the model format is Paddle, e.g Resnet50/model.pdiparams
   * \param[in] format Format of the loaded model
   */
  void SetModelPath(const std::string& model_path,
                    const std::string& params_path = "",
                    const ModelFormat& format = ModelFormat::PADDLE);

  /** \brief Specify the memory buffer of model and parameter. Used when model and params are loaded directly from memory
   *
   * \param[in] model_buffer The string of model memory buffer
   * \param[in] params_buffer The string of parameters memory buffer
   * \param[in] format Format of the loaded model
   */
  void SetModelBuffer(const std::string& model_buffer,
                      const std::string& params_buffer = "",
                      const ModelFormat& format = ModelFormat::PADDLE);

  /** \brief When loading encrypted model, encryption_key is required to decrypte model
   *
   * \param[in] encryption_key The key for decrypting model
   */
  void SetEncryptionKey(const std::string& encryption_key);

  /// Use cpu to inference, the runtime will inference on CPU by default
  void UseCpu();
  /// Use Nvidia GPU to inference
  void UseGpu(int gpu_id = 0);
  /// Use RKNPU2 e.g RK3588/RK356X to inference
  void UseRKNPU2(fastdeploy::rknpu2::CpuName rknpu2_name =
                     fastdeploy::rknpu2::CpuName::RK356X,
                 fastdeploy::rknpu2::CoreMask rknpu2_core =
                     fastdeploy::rknpu2::CoreMask::RKNN_NPU_CORE_AUTO);
  /// Use TimVX e.g RV1126/A311D to inference
  void UseTimVX();
  /// Use Huawei Ascend to inference
  void UseAscend();

  /// Use onnxruntime DirectML to inference
  void UseDirectML();

  /// Use Sophgo to inference
  void UseSophgo();
  /// \brief Turn on KunlunXin XPU.
  ///
  /// \param kunlunxin_id the KunlunXin XPU card to use (default is 0).
  /// \param l3_workspace_size The size of the video memory allocated by the l3
  ///         cache, the maximum is 16M.
  /// \param locked Whether the allocated L3 cache can be locked. If false,
  ///       it means that the L3 cache is not locked, and the allocated L3
  ///       cache can be shared by multiple models, and multiple models
  ///       sharing the L3 cache will be executed sequentially on the card.
  /// \param autotune Whether to autotune the conv operator in the model. If
  ///       true, when the conv operator of a certain dimension is executed
  ///       for the first time, it will automatically search for a better
  ///       algorithm to improve the performance of subsequent conv operators
  ///       of the same dimension.
  /// \param autotune_file Specify the path of the autotune file. If
  ///       autotune_file is specified, the algorithm specified in the
  ///       file will be used and autotune will not be performed again.
  /// \param precision Calculation accuracy of multi_encoder
  /// \param adaptive_seqlen Is the input of multi_encoder variable length
  /// \param enable_multi_stream Whether to enable the multi stream of
  ///        KunlunXin XPU.
  ///
  void UseKunlunXin(int kunlunxin_id = 0, int l3_workspace_size = 0xfffc00,
                    bool locked = false, bool autotune = true,
                    const std::string& autotune_file = "",
                    const std::string& precision = "int16",
                    bool adaptive_seqlen = false,
                    bool enable_multi_stream = false);

  void SetExternalStream(void* external_stream);

  /*
   * @brief Set number of cpu threads while inference on CPU, by default it will decided by the different backends
   */
  void SetCpuThreadNum(int thread_num);
  /// Set Paddle Inference as inference backend, support CPU/GPU
  void UsePaddleInferBackend() { return UsePaddleBackend(); }
  /// Set ONNX Runtime as inference backend, support CPU/GPU
  void UseOrtBackend();
  /// Set SOPHGO Runtime as inference backend, support SOPHGO
  void UseSophgoBackend();
  /// Set TensorRT as inference backend, only support GPU
  void UseTrtBackend();
  /// Set Poros backend as inference backend, support CPU/GPU
  void UsePorosBackend();
  /// Set OpenVINO as inference backend, only support CPU
  void UseOpenVINOBackend();
  /// Set Paddle Lite as inference backend, only support arm cpu
  void UsePaddleLiteBackend() { return UseLiteBackend(); }
  /** \Use Graphcore IPU to inference.
   *
   * \param[in] device_num the number of IPUs.
   * \param[in] micro_batch_size the batch size in the graph, only work when graph has no batch shape info.
   * \param[in] enable_pipelining enable pipelining.
   * \param[in] batches_per_step the number of batches per run in pipelining.
   */
  void UseIpu(int device_num = 1, int micro_batch_size = 1,
              bool enable_pipelining = false, int batches_per_step = 1);

  /// Option to configure ONNX Runtime backend
  OrtBackendOption ort_option;
  /// Option to configure TensorRT backend
  TrtBackendOption trt_option;
  /// Option to configure Paddle Inference backend
  PaddleBackendOption paddle_infer_option;
  /// Option to configure Poros backend
  PorosBackendOption poros_option;
  /// Option to configure OpenVINO backend
  OpenVINOBackendOption openvino_option;
  /// Option to configure Paddle Lite backend
  LiteBackendOption paddle_lite_option;
  /// Option to configure RKNPU2 backend
  RKNPU2BackendOption rknpu2_option;

  /** \brief Set the profile mode as 'true'.
   *
   * \param[in] inclue_h2d_d2h Whether to include time of H2D_D2H for time of runtime.
   * \param[in] repeat Repeat times for runtime inference.
   * \param[in] warmup Warmup times for runtime inference.
   */
  void EnableProfiling(bool inclue_h2d_d2h = false,
                       int repeat = 100, int warmup = 50) {
    benchmark_option.enable_profile = true;
    benchmark_option.warmup = warmup;
    benchmark_option.repeats = repeat;
    benchmark_option.include_h2d_d2h = inclue_h2d_d2h;
  }

  /** \brief Set the profile mode as 'false'.
   */
  void DisableProfiling() {
    benchmark_option.enable_profile = false;
  }

  /** \brief Enable to check if current backend set by user can be found at valid_xxx_backend.
   */
  void EnableValidBackendCheck() {
    enable_valid_backend_check = true;
  }
  /** \brief Disable to check if current backend set by user can be found at valid_xxx_backend.
   */
  void DisableValidBackendCheck() {
    enable_valid_backend_check = false;
  }

  /// Benchmark option
  benchmark::BenchmarkOption benchmark_option;
  // enable the check for valid backend, default true.
  bool enable_valid_backend_check = true;

  // If model_from_memory is true, the model_file and params_file is
  // binary stream in memory;
  // Otherwise, the model_file and params_file means the path of file
  std::string model_file = "";
  std::string params_file = "";
  bool model_from_memory_ = false;
  /// format of input model
  ModelFormat model_format = ModelFormat::PADDLE;

  std::string encryption_key_ = "";

  // for cpu inference
  // default will let the backend choose their own default value
  int cpu_thread_num = -1;
  int device_id = 0;
  Backend backend = Backend::UNKNOWN;

  Device device = Device::CPU;

  void* external_stream_ = nullptr;

  bool enable_pinned_memory = false;

  // *** The belowing api are deprecated, will be removed in v1.2.0
  // *** Do not use it anymore
  void SetPaddleMKLDNN(bool pd_mkldnn = true);
  void EnablePaddleToTrt();
  void DeletePaddleBackendPass(const std::string& delete_pass_name);
  void EnablePaddleLogInfo();
  void DisablePaddleLogInfo();
  void SetPaddleMKLDNNCacheSize(int size);
  void SetOpenVINODevice(const std::string& name = "CPU");
  void SetOpenVINOShapeInfo(
      const std::map<std::string, std::vector<int64_t>>& shape_info) {
    openvino_option.shape_infos = shape_info;
  }
  void SetOpenVINOCpuOperators(const std::vector<std::string>& operators) {
    openvino_option.SetCpuOperators(operators);
  }
  void SetLiteOptimizedModelDir(const std::string& optimized_model_dir);
  void SetLiteSubgraphPartitionPath(
      const std::string& nnadapter_subgraph_partition_config_path);
  void SetLiteSubgraphPartitionConfigBuffer(
      const std::string& nnadapter_subgraph_partition_config_buffer);
  void
  SetLiteContextProperties(const std::string& nnadapter_context_properties);
  void SetLiteModelCacheDir(const std::string& nnadapter_model_cache_dir);
  void SetLiteDynamicShapeInfo(
      const std::map<std::string, std::vector<std::vector<int64_t>>>&
          nnadapter_dynamic_shape_info);
  void SetLiteMixedPrecisionQuantizationConfigPath(
      const std::string& nnadapter_mixed_precision_quantization_config_path);
  void EnableLiteFP16();
  void DisableLiteFP16();
  void EnableLiteInt8();
  void DisableLiteInt8();
  void SetLitePowerMode(LitePowerMode mode);
  void SetTrtInputShape(
      const std::string& input_name, const std::vector<int32_t>& min_shape,
      const std::vector<int32_t>& opt_shape = std::vector<int32_t>(),
      const std::vector<int32_t>& max_shape = std::vector<int32_t>());
  void SetTrtMaxWorkspaceSize(size_t trt_max_workspace_size);
  void SetTrtMaxBatchSize(size_t max_batch_size);
  void EnableTrtFP16();
  void DisableTrtFP16();
  void SetTrtCacheFile(const std::string& cache_file_path);
  void EnablePinnedMemory();
  void DisablePinnedMemory();
  void EnablePaddleTrtCollectShape();
  void DisablePaddleTrtCollectShape();
  void DisablePaddleTrtOPs(const std::vector<std::string>& ops);
  void SetOpenVINOStreams(int num_streams);
  void SetOrtGraphOptLevel(int level = -1);
  void UsePaddleBackend();
  void UseLiteBackend();
};

}  // namespace fastdeploy
