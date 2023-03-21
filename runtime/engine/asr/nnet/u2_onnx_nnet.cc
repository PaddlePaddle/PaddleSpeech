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
// https://github.com/wenet-e2e/wenet/blob/main/runtime/core/decoder/asr_model.cc

#include "nnet/u2_onnx_nnet.h"

namespace ppspeech {

Ort::Env U2OnnxNnet::env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "");
Ort::SessionOptions U2OnnxNnet::session_options_ = Ort::SessionOptions();

void U2OnnxNnet::InitEngineThreads(int num_threads) {
  session_options_.SetIntraOpNumThreads(num_threads);
}

void U2OnnxNnet::LoadModel(const std::string& model_dir) {
    std::string encoder_onnx_path = model_dir + "/encoder.onnx";
    std::string rescore_onnx_path = model_dir + "/decoder.onnx";
    std::string ctc_onnx_path = model_dir + "/ctc.onnx";
    // 1. Load sessions
    try {
    #ifdef _MSC_VER
        encoder_session_ = std::make_shared<Ort::Session>(
            env_, ToWString(encoder_onnx_path).c_str(), session_options_);
        rescore_session_ = std::make_shared<Ort::Session>(
            env_, ToWString(rescore_onnx_path).c_str(), session_options_);
        ctc_session_ = std::make_shared<Ort::Session>(
            env_, ToWString(ctc_onnx_path).c_str(), session_options_);
    #else
        encoder_session_ = std::make_shared<Ort::Session>(
            env_, encoder_onnx_path.c_str(), session_options_);
        rescore_session_ = std::make_shared<Ort::Session>(
            env_, rescore_onnx_path.c_str(), session_options_);
        ctc_session_ = std::make_shared<Ort::Session>(env_, ctc_onnx_path.c_str(),
                                                    session_options_);
    #endif
    } catch (std::exception const& e) {
        LOG(ERROR) << "error when load onnx model: " << e.what();
        exit(0);
    }

    // 2. Read metadata
    auto model_metadata = encoder_session_->GetModelMetadata();

    Ort::AllocatorWithDefaultOptions allocator;
    encoder_output_size_ =
        atoi(model_metadata.LookupCustomMetadataMap("output_size", allocator));
    num_blocks_ =
        atoi(model_metadata.LookupCustomMetadataMap("num_blocks", allocator));
    head_ = atoi(model_metadata.LookupCustomMetadataMap("head", allocator));
    cnn_module_kernel_ = atoi(
        model_metadata.LookupCustomMetadataMap("cnn_module_kernel", allocator));
    subsampling_rate_ = atoi(
        model_metadata.LookupCustomMetadataMap("subsampling_rate", allocator));
    right_context_ =
        atoi(model_metadata.LookupCustomMetadataMap("right_context", allocator));
    sos_ = atoi(model_metadata.LookupCustomMetadataMap("sos_symbol", allocator));
    eos_ = atoi(model_metadata.LookupCustomMetadataMap("eos_symbol", allocator));
    is_bidecoder_ = atoi(model_metadata.LookupCustomMetadataMap(
        "is_bidirectional_decoder", allocator));
    chunk_size_ =
        atoi(model_metadata.LookupCustomMetadataMap("chunk_size", allocator));
    num_left_chunks_ =
        atoi(model_metadata.LookupCustomMetadataMap("left_chunks", allocator));

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
    GetInputOutputInfo(encoder_session_, &encoder_in_names_, &encoder_out_names_);
    LOG(INFO) << "Onnx CTC:";
    GetInputOutputInfo(ctc_session_, &ctc_in_names_, &ctc_out_names_);
    LOG(INFO) << "Onnx Rescore:";
    GetInputOutputInfo(rescore_session_, &rescore_in_names_, &rescore_out_names_);
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

    // sessions
    encoder_session_ = other.encoder_session_;
    ctc_session_ = other.ctc_session_;
    rescore_session_ = other.rescore_session_;

    // node names
    encoder_in_names_ = other.encoder_in_names_;
    encoder_out_names_ = other.encoder_out_names_;
    ctc_in_names_ = other.ctc_in_names_;
    ctc_out_names_ = other.ctc_out_names_;
    rescore_in_names_ = other.rescore_in_names_;
    rescore_out_names_ = other.rescore_out_names_;
}

void U2OnnxNnet::GetInputOutputInfo(
    const std::shared_ptr<Ort::Session>& session,
    std::vector<const char*>* in_names, std::vector<const char*>* out_names) {
  Ort::AllocatorWithDefaultOptions allocator;
  // Input info
  int num_nodes = session->GetInputCount();
  in_names->resize(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    char* name = session->GetInputName(i, allocator);
    Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::vector<int64_t> node_dims = tensor_info.GetShape();
    std::stringstream shape;
    for (auto j : node_dims) {
      shape << j;
      shape << " ";
    }
    LOG(INFO) << "\tInput " << i << " : name=" << name << " type=" << type
              << " dims=" << shape.str();
    (*in_names)[i] = name;
  }
  // Output info
  num_nodes = session->GetOutputCount();
  out_names->resize(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    char* name = session->GetOutputName(i, allocator);
    Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::vector<int64_t> node_dims = tensor_info.GetShape();
    std::stringstream shape;
    for (auto j : node_dims) {
      shape << j;
      shape << " ";
    }
    LOG(INFO) << "\tOutput " << i << " : name=" << name << " type=" << type
              << " dims=" << shape.str();
    (*out_names)[i] = name;
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
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    if (num_left_chunks_ > 0) {
        int required_cache_size = chunk_size_ * num_left_chunks_;
        offset_ = required_cache_size;
        att_cache_.resize(num_blocks_ * head_ * required_cache_size *
                            encoder_output_size_ / head_ * 2,
                        0.0);
        const int64_t att_cache_shape[] = {num_blocks_, head_, required_cache_size,
                                        encoder_output_size_ / head_ * 2};
        att_cache_ort_ = Ort::Value::CreateTensor<float>(
            memory_info, att_cache_.data(), att_cache_.size(), att_cache_shape, 4);
    } else {
        att_cache_.resize(0, 0.0);
        const int64_t att_cache_shape[] = {num_blocks_, head_, 0,
                                        encoder_output_size_ / head_ * 2};
        att_cache_ort_ = Ort::Value::CreateTensor<float>(
            memory_info, att_cache_.data(), att_cache_.size(), att_cache_shape, 4);
    }

    // Reset cnn_cache
    cnn_cache_.resize(
        num_blocks_ * encoder_output_size_ * (cnn_module_kernel_ - 1), 0.0);
    const int64_t cnn_cache_shape[] = {num_blocks_, 1, encoder_output_size_,
                                        cnn_module_kernel_ - 1};
    cnn_cache_ort_ = Ort::Value::CreateTensor<float>(
        memory_info, cnn_cache_.data(), cnn_cache_.size(), cnn_cache_shape, 4);
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
        
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    // 1. Prepare onnx required data, splice cached_feature_ and chunk_feats
    // chunk
    int num_frames = chunk_feats.size() / feat_dim;
    VLOG(3) << "num_frames: " << num_frames;
    VLOG(3) << "feat_dim: " << feat_dim;
    const int feature_dim = feat_dim;
    std::vector<float> feats;
    feats.insert(feats.end(), chunk_feats.begin(), chunk_feats.end());
    const int64_t feats_shape[3] = {1, num_frames, feature_dim};
    Ort::Value feats_ort = Ort::Value::CreateTensor<float>(
        memory_info, feats.data(), feats.size(), feats_shape, 3);
    // offset
    int64_t offset_int64 = static_cast<int64_t>(offset_);
    Ort::Value offset_ort = Ort::Value::CreateTensor<int64_t>(
        memory_info, &offset_int64, 1, std::vector<int64_t>{}.data(), 0);
    // required_cache_size
    int64_t required_cache_size = chunk_size_ * num_left_chunks_;
    Ort::Value required_cache_size_ort = Ort::Value::CreateTensor<int64_t>(
        memory_info, &required_cache_size, 1, std::vector<int64_t>{}.data(), 0);
    // att_mask
    Ort::Value att_mask_ort{nullptr};
    std::vector<uint8_t> att_mask(required_cache_size + chunk_size_, 1);
    if (num_left_chunks_ > 0) {
        int chunk_idx = offset_ / chunk_size_ - num_left_chunks_;
        if (chunk_idx < num_left_chunks_) {
        for (int i = 0; i < (num_left_chunks_ - chunk_idx) * chunk_size_; ++i) {
            att_mask[i] = 0;
        }
        }
        const int64_t att_mask_shape[] = {1, 1, required_cache_size + chunk_size_};
        att_mask_ort = Ort::Value::CreateTensor<bool>(
            memory_info, reinterpret_cast<bool*>(att_mask.data()), att_mask.size(),
            att_mask_shape, 3);
    }

    // 2. Encoder chunk forward
    std::vector<Ort::Value> inputs;
    for (auto name : encoder_in_names_) {
        if (!strcmp(name, "chunk")) {
            inputs.emplace_back(std::move(feats_ort));
        } else if (!strcmp(name, "offset")) {
            inputs.emplace_back(std::move(offset_ort));
        } else if (!strcmp(name, "required_cache_size")) {
            inputs.emplace_back(std::move(required_cache_size_ort));
        } else if (!strcmp(name, "att_cache")) {
            inputs.emplace_back(std::move(att_cache_ort_));
        } else if (!strcmp(name, "cnn_cache")) {
            inputs.emplace_back(std::move(cnn_cache_ort_));
        } else if (!strcmp(name, "att_mask")) {
            inputs.emplace_back(std::move(att_mask_ort));
        }
    }

    std::vector<Ort::Value> ort_outputs = encoder_session_->Run(
        Ort::RunOptions{nullptr}, encoder_in_names_.data(), inputs.data(),
        inputs.size(), encoder_out_names_.data(), encoder_out_names_.size());

    offset_ += static_cast<int>(
        ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1]);
    att_cache_ort_ = std::move(ort_outputs[1]);
    cnn_cache_ort_ = std::move(ort_outputs[2]);

    std::vector<Ort::Value> ctc_inputs;
    ctc_inputs.emplace_back(std::move(ort_outputs[0]));

    std::vector<Ort::Value> ctc_ort_outputs = ctc_session_->Run(
        Ort::RunOptions{nullptr}, ctc_in_names_.data(), ctc_inputs.data(),
        ctc_inputs.size(), ctc_out_names_.data(), ctc_out_names_.size());
    encoder_outs_.push_back(std::move(ctc_inputs[0]));

    float* logp_data = ctc_ort_outputs[0].GetTensorMutableData<float>();
    auto type_info = ctc_ort_outputs[0].GetTensorTypeAndShapeInfo();

    // Copy to output, (B=1,T,D)
    std::vector<int64_t> ctc_log_probs_shape = type_info.GetShape();
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
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
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
        float* encoder_outs_data = encoder_outs_[i].GetTensorMutableData<float>();
        auto type_info = encoder_outs_[i].GetTensorTypeAndShapeInfo();
        for (int j = 0; j < type_info.GetElementCount(); j++) {
        rescore_input.emplace_back(encoder_outs_data[j]);
        }
        encoder_len += type_info.GetShape()[1];
    }

    const int64_t decode_input_shape[] = {1, encoder_len, encoder_output_size_};

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

    const int64_t hyps_pad_shape[] = {num_hyps, max_hyps_len};

    const int64_t hyps_lens_shape[] = {num_hyps};

    Ort::Value decode_input_tensor_ = Ort::Value::CreateTensor<float>(
        memory_info, rescore_input.data(), rescore_input.size(),
        decode_input_shape, 3);
    Ort::Value hyps_pad_tensor_ = Ort::Value::CreateTensor<int64_t>(
        memory_info, hyps_pad.data(), hyps_pad.size(), hyps_pad_shape, 2);
    Ort::Value hyps_lens_tensor_ = Ort::Value::CreateTensor<int64_t>(
        memory_info, hyps_lens.data(), hyps_lens.size(), hyps_lens_shape, 1);

    std::vector<Ort::Value> rescore_inputs;

    rescore_inputs.emplace_back(std::move(hyps_pad_tensor_));
    rescore_inputs.emplace_back(std::move(hyps_lens_tensor_));
    rescore_inputs.emplace_back(std::move(decode_input_tensor_));

    std::vector<Ort::Value> rescore_outputs = rescore_session_->Run(
        Ort::RunOptions{nullptr}, rescore_in_names_.data(), rescore_inputs.data(),
        rescore_inputs.size(), rescore_out_names_.data(),
        rescore_out_names_.size());

    float* decoder_outs_data = rescore_outputs[0].GetTensorMutableData<float>();
    float* r_decoder_outs_data = rescore_outputs[1].GetTensorMutableData<float>();

    auto type_info = rescore_outputs[0].GetTensorTypeAndShapeInfo();
    int decode_out_len = type_info.GetShape()[2];

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