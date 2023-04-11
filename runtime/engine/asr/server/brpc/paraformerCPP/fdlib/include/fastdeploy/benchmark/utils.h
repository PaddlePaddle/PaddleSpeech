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

#include <memory>
#include <thread>  // NOLINT
#include "fastdeploy/utils/utils.h"
#include "fastdeploy/core/fd_tensor.h"
#if defined(ENABLE_BENCHMARK) && defined(ENABLE_VISION)
#include "fastdeploy/vision/common/result.h"
#endif

namespace fastdeploy {
namespace benchmark {

#if defined(ENABLE_BENCHMARK)
/*! @brief ResourceUsageMonitor object used when to collect memory info.
 */
class FASTDEPLOY_DECL ResourceUsageMonitor {
 public:
   /** \brief  Set sampling_interval_ms and gpu_id for ResourceUsageMonitor.
   *
   * \param[in] sampling_interval_ms How often to collect memory info(ms).
   * \param[in] gpu_id Device(gpu) id, default 0.
   */
  explicit ResourceUsageMonitor(int sampling_interval_ms, int gpu_id = 0);

  ~ResourceUsageMonitor() { StopInternal(); }

  /// Start memory info collect
  void Start();
  /// Stop memory info collect
  void Stop();
  /// Get maximum cpu memory usage
  float GetMaxCpuMem() const {
    if (!is_supported_ || check_memory_thd_ == nullptr) {
      return -1.0f;
    }
    return max_cpu_mem_;
  }
  /// Get maximum gpu memory usage
  float GetMaxGpuMem() const {
    if (!is_supported_ || check_memory_thd_ == nullptr) {
      return -1.0f;
    }
    return max_gpu_mem_;
  }
  /// Get maximum gpu util
  float GetMaxGpuUtil() const {
    if (!is_supported_ || check_memory_thd_ == nullptr) {
      return -1.0f;
    }
    return max_gpu_util_;
  }

  ResourceUsageMonitor(ResourceUsageMonitor&) = delete;
  ResourceUsageMonitor& operator=(const ResourceUsageMonitor&) = delete;
  ResourceUsageMonitor(ResourceUsageMonitor&&) = delete;
  ResourceUsageMonitor& operator=(const ResourceUsageMonitor&&) = delete;

 private:
  void StopInternal();
  // Get current gpu memory info
  std::string GetCurrentGpuMemoryInfo(int device_id);

  bool is_supported_ = false;
  bool stop_signal_ = false;
  const int sampling_interval_;
  float max_cpu_mem_ = 0.0f;  // MB
  float max_gpu_mem_ = 0.0f;  // MB
  float max_gpu_util_ = 0.0f;
  const int gpu_id_ = 0;
  std::unique_ptr<std::thread> check_memory_thd_ = nullptr;
};

// Remove the ch characters at both ends of str
FASTDEPLOY_DECL std::string Strip(const std::string& str, char ch = ' ');

// Split string
FASTDEPLOY_DECL void Split(const std::string& s,
                           std::vector<std::string>& tokens,
                           char delim = ' ');

/// Diff values for precision evaluation
struct FASTDEPLOY_DECL BaseDiff {};

struct FASTDEPLOY_DECL EvalStatis {
  double mean = -1.0;
  double min = -1.0;
  double max = -1.0;
};

struct FASTDEPLOY_DECL TensorDiff: public BaseDiff {
  EvalStatis data;
};

#if defined(ENABLE_VISION)
struct FASTDEPLOY_DECL DetectionDiff: public BaseDiff {
  EvalStatis boxes;
  EvalStatis scores;
  EvalStatis labels;
};

struct FASTDEPLOY_DECL ClassifyDiff: public BaseDiff {
  EvalStatis scores;
  EvalStatis labels;
};

struct FASTDEPLOY_DECL SegmentationDiff: public BaseDiff {
  EvalStatis scores;
  EvalStatis labels;
};

struct FASTDEPLOY_DECL OCRDetDiff: public BaseDiff {
  EvalStatis boxes;
};

#endif  // ENABLE_VISION
#endif  // ENABLE_BENCHMARK

/// Utils for precision evaluation
struct FASTDEPLOY_DECL ResultManager {
#if defined(ENABLE_BENCHMARK)
  /// Save & Load functions for FDTensor result.
  static bool SaveFDTensor(const FDTensor& tensor, const std::string& path);
  static bool LoadFDTensor(FDTensor* tensor, const std::string& path);
  /// Calculate diff value between two FDTensor results.
  static TensorDiff CalculateDiffStatis(const FDTensor& lhs,
                                        const FDTensor& rhs);
  /// Save Benchmark data
  static void SaveBenchmarkResult(const std::string& res,
                                  const std::string& path);
#if defined(ENABLE_VISION)
  /// Save & Load functions for basic results.
  static bool SaveDetectionResult(const vision::DetectionResult& res,
                                  const std::string& path);
  static bool LoadDetectionResult(vision::DetectionResult* res,
                                  const std::string& path);
  static bool SaveClassifyResult(const vision::ClassifyResult& res,
                                 const std::string& path);
  static bool LoadClassifyResult(vision::ClassifyResult* res,
                                 const std::string& path);
  static bool SaveSegmentationResult(const vision::SegmentationResult& res,
                                     const std::string& path);
  static bool LoadSegmentationResult(vision::SegmentationResult* res,
                                     const std::string& path);
  static bool SaveOCRDetResult(const std::vector<std::array<int, 8>>& res,
                               const std::string& path);
  static bool LoadOCRDetResult(std::vector<std::array<int, 8>>* res,
                               const std::string& path);
  /// Calculate diff value between two basic results.
  static DetectionDiff CalculateDiffStatis(const vision::DetectionResult& lhs,
                                           const vision::DetectionResult& rhs,
                                           const float& score_threshold = 0.3f);
  static ClassifyDiff CalculateDiffStatis(const vision::ClassifyResult& lhs,
                                          const vision::ClassifyResult& rhs);
  static SegmentationDiff CalculateDiffStatis(
      const vision::SegmentationResult& lhs,
      const vision::SegmentationResult& rhs);
  static OCRDetDiff CalculateDiffStatis(
      const std::vector<std::array<int, 8>>& lhs,
      const std::vector<std::array<int, 8>>& rhs);
#endif  // ENABLE_VISION
#endif  // ENABLE_BENCHMARK
};

}  // namespace benchmark
}  // namespace fastdeploy
