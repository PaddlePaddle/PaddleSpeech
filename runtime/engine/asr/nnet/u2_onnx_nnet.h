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

#include "fastdeploy/runtime.h"

namespace ppspeech {

class U2OnnxNnet : public U2NnetBase {

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

    std::shared_ptr<NnetBase> Clone() const override;

    void ForwardEncoderChunkImpl(
        const std::vector<kaldi::BaseFloat>& chunk_feats,
        const int32& feat_dim,
        std::vector<kaldi::BaseFloat>* ctc_probs,
        int32* vocab_dim) override;

    float ComputeAttentionScore(const float* prob, const std::vector<int>& hyp,
                              int eos, int decode_out_len);

    void AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                            float reverse_weight,
                            std::vector<float>* rescoring_score) override;

    void EncoderOuts(
        std::vector<std::vector<kaldi::BaseFloat>>* encoder_out) const;

    void GetInputOutputInfo(const std::shared_ptr<fastdeploy::Runtime>& runtime,
                          std::vector<std::string>* in_names,
                          std::vector<std::string>* out_names);
  private:
    ModelOptions opts_;

    int encoder_output_size_ = 0;
    int num_blocks_ = 0;
    int cnn_module_kernel_ = 0;
    int head_ = 0;

    // sessions
    std::shared_ptr<fastdeploy::Runtime> encoder_ = nullptr;
    std::shared_ptr<fastdeploy::Runtime> rescore_ = nullptr;
    std::shared_ptr<fastdeploy::Runtime> ctc_ = nullptr;


    // node names
    std::vector<std::string> encoder_in_names_, encoder_out_names_;
    std::vector<std::string> ctc_in_names_, ctc_out_names_;
    std::vector<std::string> rescore_in_names_, rescore_out_names_;

    // caches
    fastdeploy::FDTensor att_cache_ort_;
    fastdeploy::FDTensor cnn_cache_ort_;
    std::vector<fastdeploy::FDTensor> encoder_outs_;

    std::vector<float> att_cache_;
    std::vector<float> cnn_cache_;
};

}  // namespace ppspeech