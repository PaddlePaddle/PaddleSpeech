// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang, Di Wu)
//               2022 ZeXuan Li (lizexuan@huya.com)
//                    Xingchen Song(sxc19@mails.tsinghua.edu.cn)
//                    hamddct@gmail.com (Mddct)
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

#include "base/common.h"
#include "matrix/kaldi-matrix.h"
#include "nnet/nnet_itf.h"
#include "nnet/u2_nnet.h"

#include "onnxruntime_cxx_api.h"  // NOLINT


namespace ppspeech {

class U2OnnxNnet : public U2NnetBase {
  public:
    static void InitEngineThreads(int num_threads = 1);

  public:
    explicit U2OnnxNnet(const ModelOptions& opts);
    U2OnnxNnet(const U2OnnxNnet& other);

    void FeedForward(const std::vector<kaldi::BaseFloat>& features,
                     const int32& feature_dim,
                     NnetOut* out) override;

    void Reset() override;

    bool IsLogProb() override { return true; }

    void Dim();

    void LoadModel(const std::string& model_dir);
    // void Warmup();

    // std::shared_ptr<paddle::jit::Layer> model() const { return model_; }

    std::shared_ptr<NnetBase> Clone() const override;

    void ForwardEncoderChunkImpl(
        const std::vector<kaldi::BaseFloat>& chunk_feats,
        const int32& feat_dim,
        std::vector<kaldi::BaseFloat>* ctc_probs,
        int32* vocab_dim) override;

    // float ComputePathScore(const paddle::Tensor& prob,
    //                        const std::vector<int>& hyp,
    //                        int eos);
    float ComputeAttentionScore(const float* prob, const std::vector<int>& hyp,
                              int eos, int decode_out_len);

    void AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                            float reverse_weight,
                            std::vector<float>* rescoring_score) override;

    // debug
    // void FeedEncoderOuts(const paddle::Tensor& encoder_out);

    void EncoderOuts(
        std::vector<std::vector<kaldi::BaseFloat>>* encoder_out) const;

    // copy from wenet
    void GetInputOutputInfo(const std::shared_ptr<Ort::Session>& session,
                          std::vector<const char*>* in_names,
                          std::vector<const char*>* out_names);
  private:
    ModelOptions opts_;

    int encoder_output_size_ = 0;
    int num_blocks_ = 0;
    int cnn_module_kernel_ = 0;
    int head_ = 0;

    // sessions
    // NOTE(Mddct): The Env holds the logging state used by all other objects.
    //  One Env must be created before using any other Onnxruntime functionality.
    static Ort::Env env_;  // shared environment across threads.
    static Ort::SessionOptions session_options_;
    std::shared_ptr<Ort::Session> encoder_session_ = nullptr;
    std::shared_ptr<Ort::Session> rescore_session_ = nullptr;
    std::shared_ptr<Ort::Session> ctc_session_ = nullptr;

    // node names
    std::vector<const char*> encoder_in_names_, encoder_out_names_;
    std::vector<const char*> ctc_in_names_, ctc_out_names_;
    std::vector<const char*> rescore_in_names_, rescore_out_names_;

    // caches
    Ort::Value att_cache_ort_{nullptr};
    Ort::Value cnn_cache_ort_{nullptr};
    std::vector<Ort::Value> encoder_outs_;
    // NOTE: Instead of making a copy of the xx_cache, ONNX only maintains
    //  its data pointer when initializing xx_cache_ort (see https://github.com/
    //  microsoft/onnxruntime/blob/master/onnxruntime/core/framework
    //  /tensor.cc#L102-L129), so we need the following variables to keep
    //  our data "alive" during the lifetime of decoder.
    std::vector<float> att_cache_;
    std::vector<float> cnn_cache_;
};

}  // namespace ppspeech