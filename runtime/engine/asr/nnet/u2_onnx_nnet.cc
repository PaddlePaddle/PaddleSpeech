// Copyright 2022 Horizon Robotics. All Rights Reserved.
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

// modified from
// https://github.com/wenet-e2e/wenet/blob/main/runtime/core/decoder/onnx_asr_model.cc

#include "nnet/u2_onnx_nnet.h"
#include "common/base/config.h"

namespace ppspeech {

void U2OnnxNnet::LoadModel(const std::string& model_dir) {
    std::string encoder_onnx_path = model_dir + "/encoder.onnx";
    std::string rescore_onnx_path = model_dir + "/decoder.onnx";
    std::string ctc_onnx_path = model_dir + "/ctc.onnx";
    std::string param_path = model_dir + "/param.onnx";
    // 1. Load sessions
    try {
        encoder_ = std::make_shared<fastdeploy::Runtime>();
        ctc_ = std::make_shared<fastdeploy::Runtime>();
        rescore_ = std::make_shared<fastdeploy::Runtime>();
        fastdeploy::RuntimeOption runtime_option;
        runtime_option.UseOrtBackend();
        runtime_option.UseCpu();
        runtime_option.SetCpuThreadNum(1);
        runtime_option.SetModelPath(encoder_onnx_path.c_str(), "", fastdeploy::ModelFormat::ONNX);
        assert(encoder_->Init(runtime_option));
        runtime_option.SetModelPath(rescore_onnx_path.c_str(), "", fastdeploy::ModelFormat::ONNX);
        assert(rescore_->Init(runtime_option));
        runtime_option.SetModelPath(ctc_onnx_path.c_str(), "", fastdeploy::ModelFormat::ONNX);
        assert(ctc_->Init(runtime_option));
    } catch (std::exception const& e) {
        LOG(ERROR) << "error when load onnx model: " << e.what();
        exit(0);
    }

    Config conf(param_path);
    encoder_output_size_ = conf.Read("output_size", encoder_output_size_);
    num_blocks_ = conf.Read("num_blocks", num_blocks_);
    head_ = conf.Read("head", head_);
    cnn_module_kernel_ = conf.Read("cnn_module_kernel", cnn_module_kernel_);
    subsampling_rate_ = conf.Read("subsampling_rate", subsampling_rate_);
    right_context_ = conf.Read("right_context", right_context_);
    sos_= conf.Read("sos_symbol", sos_);
    eos_= conf.Read("eos_symbol", eos_);
    is_bidecoder_= conf.Read("is_bidirectional_decoder", is_bidecoder_);
    chunk_size_= conf.Read("chunk_size", chunk_size_);
    num_left_chunks_ = conf.Read("left_chunks", num_left_chunks_);
    
    LOG(INFO) << "Onnx Model Info:";
    LOG(INFO) << "\tencoder_output_size " << encoder_output_size_;
    LOG(INFO) << "\tnum_blocks " << num_blocks_;
    LOG(INFO) << "\thead " << head_;
    LOG(INFO) << "\tcnn_module_kernel " << cnn_module_kernel_;
    LOG(INFO) << "\tsubsampling_rate " << subsampling_rate_;
    LOG(INFO) << "\tright_context " << right_context_;
    LOG(INFO) << "\tsos " << sos_;
    LOG(INFO) << "\teos " << eos_;
    LOG(INFO) << "\tis bidirectional decoder " << is_bidecoder_;
    LOG(INFO) << "\tchunk_size " << chunk_size_;
    LOG(INFO) << "\tnum_left_chunks " << num_left_chunks_;

    // 3. Read model nodes
    LOG(INFO) << "Onnx Encoder:";
    GetInputOutputInfo(encoder_, &encoder_in_names_, &encoder_out_names_);
    LOG(INFO) << "Onnx CTC:";
    GetInputOutputInfo(ctc_, &ctc_in_names_, &ctc_out_names_);
    LOG(INFO) << "Onnx Rescore:";
    GetInputOutputInfo(rescore_, &rescore_in_names_, &rescore_out_names_);
}

U2OnnxNnet::U2OnnxNnet(const ModelOptions& opts) : opts_(opts) {
    LoadModel(opts_.model_path);
}

// shallow copy
U2OnnxNnet::U2OnnxNnet(const U2OnnxNnet& other) {
    // metadatas
    encoder_output_size_ = other.encoder_output_size_;
    num_blocks_ = other.num_blocks_;
    head_ = other.head_;
    cnn_module_kernel_ = other.cnn_module_kernel_;
    right_context_ = other.right_context_;
    subsampling_rate_ = other.subsampling_rate_;
    sos_ = other.sos_;
    eos_ = other.eos_;
    is_bidecoder_ = other.is_bidecoder_;
    chunk_size_ = other.chunk_size_;
    num_left_chunks_ = other.num_left_chunks_;
    offset_ = other.offset_;
    
    // session
    encoder_ = other.encoder_;
    ctc_ = other.ctc_;
    rescore_ = other.rescore_;

    // node names
    encoder_in_names_ = other.encoder_in_names_;
    encoder_out_names_ = other.encoder_out_names_;
    ctc_in_names_ = other.ctc_in_names_;
    ctc_out_names_ = other.ctc_out_names_;
    rescore_in_names_ = other.rescore_in_names_;
    rescore_out_names_ = other.rescore_out_names_;
}

void U2OnnxNnet::GetInputOutputInfo(const std::shared_ptr<fastdeploy::Runtime>& runtime,
                                std::vector<std::string>* in_names, std::vector<std::string>* out_names) {
    std::vector<fastdeploy::TensorInfo> inputs_info = runtime->GetInputInfos();
    (*in_names).resize(inputs_info.size());
    for (int i = 0; i < inputs_info.size(); ++i){
        fastdeploy::TensorInfo info = inputs_info[i];

        std::stringstream shape;
        for(int j = 0; j < info.shape.size(); ++j){
            shape << info.shape[j];
            shape << " ";
        }
        LOG(INFO) << "\tInput " << i << " : name=" << info.name << " type=" << info.dtype
              << " dims=" << shape.str();
        (*in_names)[i] = info.name;
    }
    std::vector<fastdeploy::TensorInfo> outputs_info = runtime->GetOutputInfos();
    (*out_names).resize(outputs_info.size());
    for (int i = 0; i < outputs_info.size(); ++i){
        fastdeploy::TensorInfo info = outputs_info[i];
        
        std::stringstream shape;
        for(int j = 0; j < info.shape.size(); ++j){
            shape << info.shape[j];
            shape << " ";
        }
        LOG(INFO) << "\tOutput " << i << " : name=" << info.name << " type=" << info.dtype
              << " dims=" << shape.str();
        (*out_names)[i] = info.name;
    }
}

std::shared_ptr<NnetBase> U2OnnxNnet::Clone() const {
    auto asr_model = std::make_shared<U2OnnxNnet>(*this);
    // reset inner state for new decoding
    asr_model->Reset();
    return asr_model;
}

void U2OnnxNnet::Reset() {
    offset_ = 0;
    encoder_outs_.clear();
    cached_feats_.clear();
    // Reset att_cache
    if (num_left_chunks_ > 0) {
        int required_cache_size = chunk_size_ * num_left_chunks_;
        offset_ = required_cache_size;
        att_cache_.resize(num_blocks_ * head_ * required_cache_size *
                            encoder_output_size_ / head_ * 2,
                        0.0);
        const std::vector<int64_t> att_cache_shape = {num_blocks_, head_, required_cache_size,
                                        encoder_output_size_ / head_ * 2};
        att_cache_ort_.SetExternalData(att_cache_shape, fastdeploy::FDDataType::FP32, att_cache_.data());
    } else {
        att_cache_.resize(0, 0.0);
        const std::vector<int64_t> att_cache_shape = {num_blocks_, head_, 0,
                                        encoder_output_size_ / head_ * 2};
        att_cache_ort_.SetExternalData(att_cache_shape, fastdeploy::FDDataType::FP32, att_cache_.data());
    }

    // Reset cnn_cache
    cnn_cache_.resize(
        num_blocks_ * encoder_output_size_ * (cnn_module_kernel_ - 1), 0.0);
    const std::vector<int64_t> cnn_cache_shape = {num_blocks_, 1, encoder_output_size_,
                                        cnn_module_kernel_ - 1};
    cnn_cache_ort_.SetExternalData(cnn_cache_shape, fastdeploy::FDDataType::FP32, cnn_cache_.data());
}

void U2OnnxNnet::FeedForward(const std::vector<BaseFloat>& features,
                         const int32& feature_dim,
                         NnetOut* out) {
    kaldi::Timer timer;

    std::vector<kaldi::BaseFloat> ctc_probs;
    ForwardEncoderChunkImpl(
        features, feature_dim, &out->logprobs, &out->vocab_dim);
    VLOG(1) << "FeedForward cost: " << timer.Elapsed() << " sec. "
            << features.size() / feature_dim << " frames.";
}

void U2OnnxNnet::ForwardEncoderChunkImpl(
        const std::vector<kaldi::BaseFloat>& chunk_feats,
        const int32& feat_dim,
        std::vector<kaldi::BaseFloat>* out_prob,
        int32* vocab_dim) {
        
    // 1. Prepare onnx required data, splice cached_feature_ and chunk_feats
    // chunk
    int num_frames = chunk_feats.size() / feat_dim;
    VLOG(3) << "num_frames: " << num_frames;
    VLOG(3) << "feat_dim: " << feat_dim;
    const int feature_dim = feat_dim;
    std::vector<float> feats;
    feats.insert(feats.end(), chunk_feats.begin(), chunk_feats.end());
    fastdeploy::FDTensor feats_ort;
    const std::vector<int64_t> feats_shape = {1, num_frames, feature_dim};
    feats_ort.SetExternalData(feats_shape, fastdeploy::FDDataType::FP32, feats.data());

    // offset
    int64_t offset_int64 = static_cast<int64_t>(offset_);
    fastdeploy::FDTensor offset_ort;
    offset_ort.SetExternalData({}, fastdeploy::FDDataType::INT64, &offset_int64);

    // required_cache_size
    int64_t required_cache_size = chunk_size_ * num_left_chunks_;
    fastdeploy::FDTensor required_cache_size_ort("");
    required_cache_size_ort.SetExternalData({}, fastdeploy::FDDataType::INT64, &required_cache_size);

    // att_mask
    fastdeploy::FDTensor att_mask_ort;
    std::vector<uint8_t> att_mask(required_cache_size + chunk_size_, 1);
    if (num_left_chunks_ > 0) {
        int chunk_idx = offset_ / chunk_size_ - num_left_chunks_;
        if (chunk_idx < num_left_chunks_) {
            for (int i = 0; i < (num_left_chunks_ - chunk_idx) * chunk_size_; ++i) {
                att_mask[i] = 0;
            }
        }
        const std::vector<int64_t> att_mask_shape = {1, 1, required_cache_size + chunk_size_};
        att_mask_ort.SetExternalData(att_mask_shape, fastdeploy::FDDataType::BOOL, reinterpret_cast<bool*>(att_mask.data()));
    }

    // 2. Encoder chunk forward
    std::vector<fastdeploy::FDTensor> inputs(encoder_in_names_.size());
    for (int i = 0; i < encoder_in_names_.size(); ++i) {
        std::string name = encoder_in_names_[i];
        if (!strcmp(name.data(), "chunk")) {
            inputs[i] = std::move(feats_ort);
            inputs[i].name = "chunk";
        } else if (!strcmp(name.data(), "offset")) {
            inputs[i] = std::move(offset_ort);
            inputs[i].name = "offset";
        } else if (!strcmp(name.data(), "required_cache_size")) {
            inputs[i] = std::move(required_cache_size_ort);
            inputs[i].name = "required_cache_size";
        } else if (!strcmp(name.data(), "att_cache")) {
            inputs[i] = std::move(att_cache_ort_);
            inputs[i].name = "att_cache";
        } else if (!strcmp(name.data(), "cnn_cache")) {
            inputs[i] = std::move(cnn_cache_ort_);
            inputs[i].name = "cnn_cache";
        } else if (!strcmp(name.data(), "att_mask")) {
            inputs[i] = std::move(att_mask_ort);
            inputs[i].name = "att_mask";
        }
    }
   
    std::vector<fastdeploy::FDTensor> ort_outputs;
    assert(encoder_->Infer(inputs, &ort_outputs));

    offset_ += static_cast<int>(ort_outputs[0].shape[1]);
    att_cache_ort_ = std::move(ort_outputs[1]);
    cnn_cache_ort_ = std::move(ort_outputs[2]);

    std::vector<fastdeploy::FDTensor> ctc_inputs;
    ctc_inputs.emplace_back(std::move(ort_outputs[0]));
    // ctc_inputs[0] = std::move(ort_outputs[0]);
    ctc_inputs[0].name = ctc_in_names_[0];

    std::vector<fastdeploy::FDTensor> ctc_ort_outputs;
    assert(ctc_->Infer(ctc_inputs, &ctc_ort_outputs));
    encoder_outs_.emplace_back(std::move(ctc_inputs[0])); // *****

    float* logp_data = reinterpret_cast<float*>(ctc_ort_outputs[0].Data());

    // Copy to output, (B=1,T,D)
    std::vector<int64_t> ctc_log_probs_shape = ctc_ort_outputs[0].shape;
    CHECK_EQ(ctc_log_probs_shape.size(), 3);
    int B = ctc_log_probs_shape[0];
    CHECK_EQ(B, 1);
    int T = ctc_log_probs_shape[1];
    int D = ctc_log_probs_shape[2];
    *vocab_dim = D;

    out_prob->resize(T * D);
    std::memcpy(
        out_prob->data(), logp_data, T * D * sizeof(kaldi::BaseFloat));
    return;
}

float U2OnnxNnet::ComputeAttentionScore(const float* prob,
                                          const std::vector<int>& hyp, int eos,
                                          int decode_out_len) {
  float score = 0.0f;
  for (size_t j = 0; j < hyp.size(); ++j) {
    score += *(prob + j * decode_out_len + hyp[j]);
  }
  score += *(prob + hyp.size() * decode_out_len + eos);
  return score;
}

void U2OnnxNnet::AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                                float reverse_weight,
                                std::vector<float>* rescoring_score) {
    CHECK(rescoring_score != nullptr);
    int num_hyps = hyps.size();
    rescoring_score->resize(num_hyps, 0.0f);

    if (num_hyps == 0) {
        return;
    }
    // No encoder output
    if (encoder_outs_.size() == 0) {
        return;
    }

    std::vector<int64_t> hyps_lens;
    int max_hyps_len = 0;
    for (size_t i = 0; i < num_hyps; ++i) {
        int length = hyps[i].size() + 1;
        max_hyps_len = std::max(length, max_hyps_len);
        hyps_lens.emplace_back(static_cast<int64_t>(length));
    }

    std::vector<float> rescore_input;
    int encoder_len = 0;
    for (int i = 0; i < encoder_outs_.size(); i++) {
        float* encoder_outs_data = reinterpret_cast<float*>(encoder_outs_[i].Data());
        for (int j = 0; j < encoder_outs_[i].Numel(); j++) {
            rescore_input.emplace_back(encoder_outs_data[j]);
        }
        encoder_len += encoder_outs_[i].shape[1];
    }

    std::vector<int64_t> hyps_pad;

    for (size_t i = 0; i < num_hyps; ++i) {
        const std::vector<int>& hyp = hyps[i];
        hyps_pad.emplace_back(sos_);
        size_t j = 0;
        for (; j < hyp.size(); ++j) {
            hyps_pad.emplace_back(hyp[j]);
        }
        if (j == max_hyps_len - 1) {
            continue;
        }
        for (; j < max_hyps_len - 1; ++j) {
            hyps_pad.emplace_back(0);
        }
    }

    const std::vector<int64_t> hyps_pad_shape = {num_hyps, max_hyps_len};
    const std::vector<int64_t> hyps_lens_shape = {num_hyps};
    const std::vector<int64_t> decode_input_shape = {1, encoder_len, encoder_output_size_};

    fastdeploy::FDTensor hyps_pad_tensor_;
    hyps_pad_tensor_.SetExternalData(hyps_pad_shape, fastdeploy::FDDataType::INT64, hyps_pad.data());
    fastdeploy::FDTensor hyps_lens_tensor_;
    hyps_lens_tensor_.SetExternalData(hyps_lens_shape, fastdeploy::FDDataType::INT64, hyps_lens.data());
    fastdeploy::FDTensor decode_input_tensor_;
    decode_input_tensor_.SetExternalData(decode_input_shape, fastdeploy::FDDataType::FP32, rescore_input.data());

    std::vector<fastdeploy::FDTensor> rescore_inputs(3);

    rescore_inputs[0] = std::move(hyps_pad_tensor_);
    rescore_inputs[0].name = rescore_in_names_[0];
    rescore_inputs[1] = std::move(hyps_lens_tensor_);
    rescore_inputs[1].name = rescore_in_names_[1];
    rescore_inputs[2] = std::move(decode_input_tensor_);
    rescore_inputs[2].name = rescore_in_names_[2];

    std::vector<fastdeploy::FDTensor> rescore_outputs;
    assert(rescore_->Infer(rescore_inputs, &rescore_outputs));

    float* decoder_outs_data = reinterpret_cast<float*>(rescore_outputs[0].Data());
    float* r_decoder_outs_data = reinterpret_cast<float*>(rescore_outputs[1].Data());

    int decode_out_len = rescore_outputs[0].shape[2];

    for (size_t i = 0; i < num_hyps; ++i) {
        const std::vector<int>& hyp = hyps[i];
        float score = 0.0f;
        // left to right decoder score
        score = ComputeAttentionScore(
            decoder_outs_data + max_hyps_len * decode_out_len * i, hyp, eos_,
            decode_out_len);
        // Optional: Used for right to left score
        float r_score = 0.0f;
        if (is_bidecoder_ && reverse_weight > 0) {
        std::vector<int> r_hyp(hyp.size());
        std::reverse_copy(hyp.begin(), hyp.end(), r_hyp.begin());
        // right to left decoder score
        r_score = ComputeAttentionScore(
            r_decoder_outs_data + max_hyps_len * decode_out_len * i, r_hyp, eos_,
            decode_out_len);
        }
        // combined left-to-right and right-to-left score
        (*rescoring_score)[i] =
            score * (1 - reverse_weight) + r_score * reverse_weight;
    }
}

void U2OnnxNnet::EncoderOuts(
    std::vector<std::vector<kaldi::BaseFloat>>* encoder_out) const {
}

} //namepace ppspeech