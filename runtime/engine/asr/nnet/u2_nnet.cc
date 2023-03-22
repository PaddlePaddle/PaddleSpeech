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

#include "nnet/u2_nnet.h"

#ifdef WITH_PROFILING
#include "paddle/fluid/platform/profiler.h"
using paddle::platform::RecordEvent;
using paddle::platform::TracerEventType;
#endif  // end WITH_PROFILING

namespace ppspeech {


void U2Nnet::LoadModel(const std::string& model_path_w_prefix) {
    paddle::jit::utils::InitKernelSignatureMap();

#ifdef WITH_GPU
    dev_ = phi::GPUPlace();
#else
    dev_ = phi::CPUPlace();
#endif
    paddle::jit::Layer model = paddle::jit::Load(model_path_w_prefix, dev_);
    model_ = std::make_shared<paddle::jit::Layer>(std::move(model));

    subsampling_rate_ = model_->Attribute<int>("subsampling_rate");
    right_context_ = model_->Attribute<int>("right_context");
    sos_ = model_->Attribute<int>("sos_symbol");
    eos_ = model_->Attribute<int>("eos_symbol");
    is_bidecoder_ = model_->Attribute<int>("is_bidirectional_decoder");

    forward_encoder_chunk_ = model_->Function("forward_encoder_chunk");
    forward_attention_decoder_ = model_->Function("forward_attention_decoder");
    ctc_activation_ = model_->Function("ctc_activation");
    CHECK(forward_encoder_chunk_.IsValid());
    CHECK(forward_attention_decoder_.IsValid());
    CHECK(ctc_activation_.IsValid());

    LOG(INFO) << "Paddle Model Info: ";
    LOG(INFO) << "\tsubsampling_rate " << subsampling_rate_;
    LOG(INFO) << "\tright context " << right_context_;
    LOG(INFO) << "\tsos " << sos_;
    LOG(INFO) << "\teos " << eos_;
    LOG(INFO) << "\tis bidecoder " << is_bidecoder_ << std::endl;

    Warmup();
}

void U2Nnet::Warmup() {
#ifdef WITH_PROFILING
    RecordEvent event("warmup", TracerEventType::UserDefined, 1);
#endif

    {
#ifdef WITH_PROFILING
        RecordEvent event(
            "warmup-encoder-ctc", TracerEventType::UserDefined, 1);
#endif
        int feat_dim = 80;
        int frame_num = 16 * 4 + 3;  // chunk_size * downsample_rate +
                                     // (receptive_field - downsample_rate)
        paddle::Tensor feats = paddle::full(
            {1, frame_num, feat_dim}, 0.12f, paddle::DataType::FLOAT32);
        paddle::Tensor offset = paddle::zeros({1}, paddle::DataType::INT32);
        paddle::Tensor att_cache =
            paddle::zeros({0, 0, 0, 0}, paddle::DataType::FLOAT32);
        paddle::Tensor cnn_cache =
            paddle::zeros({0, 0, 0, 0}, paddle::DataType::FLOAT32);
        std::vector<paddle::Tensor> inputs = {
            feats, offset, /*required_cache_size, */ att_cache, cnn_cache};
        std::vector<paddle::Tensor> outputs = forward_encoder_chunk_(inputs);

        auto chunk_out = outputs[0];
        inputs = std::move(std::vector<paddle::Tensor>({chunk_out}));
        outputs = ctc_activation_(inputs);
    }

    {
#ifdef WITH_PROFILING
        RecordEvent event("warmup-decoder", TracerEventType::UserDefined, 1);
#endif
        auto hyps =
            paddle::full({10, 8}, 10, paddle::DataType::INT64, phi::CPUPlace());
        auto hyps_lens =
            paddle::full({10}, 8, paddle::DataType::INT64, phi::CPUPlace());
        auto encoder_out = paddle::ones(
            {1, 20, 512}, paddle::DataType::FLOAT32, phi::CPUPlace());

        std::vector<paddle::Tensor> inputs{
            hyps, hyps_lens, encoder_out};

        std::vector<paddle::Tensor> outputs =
            forward_attention_decoder_(inputs);
    }

    Reset();
}

U2Nnet::U2Nnet(const ModelOptions& opts) : opts_(opts) {
    LoadModel(opts_.model_path);
}

// shallow copy
U2Nnet::U2Nnet(const U2Nnet& other) {
    // copy meta
    chunk_size_ = other.chunk_size_;
    num_left_chunks_ = other.num_left_chunks_;
    offset_ = other.offset_;

    // copy model ptr
    model_ = other.model_->Clone();
    ctc_activation_ = model_->Function("ctc_activation");
    subsampling_rate_ = model_->Attribute<int>("subsampling_rate");
    right_context_ = model_->Attribute<int>("right_context");
    sos_ = model_->Attribute<int>("sos_symbol");
    eos_ = model_->Attribute<int>("eos_symbol");
    is_bidecoder_ = model_->Attribute<int>("is_bidirectional_decoder");

    forward_encoder_chunk_ = model_->Function("forward_encoder_chunk");
    forward_attention_decoder_ = model_->Function("forward_attention_decoder");
    ctc_activation_ = model_->Function("ctc_activation");
    CHECK(forward_encoder_chunk_.IsValid());
    CHECK(forward_attention_decoder_.IsValid());
    CHECK(ctc_activation_.IsValid());

    LOG(INFO) << "Paddle Model Info: ";
    LOG(INFO) << "\tsubsampling_rate " << subsampling_rate_;
    LOG(INFO) << "\tright context " << right_context_;
    LOG(INFO) << "\tsos " << sos_;
    LOG(INFO) << "\teos " << eos_;
    LOG(INFO) << "\tis bidecoder " << is_bidecoder_ << std::endl;


    // ignore inner states
}

std::shared_ptr<NnetBase> U2Nnet::Clone() const {
    auto asr_model = std::make_shared<U2Nnet>(*this);
    // reset inner state for new decoding
    asr_model->Reset();
    return asr_model;
}

void U2Nnet::Reset() {
    offset_ = 0;

    att_cache_ =
        std::move(paddle::zeros({0, 0, 0, 0}, paddle::DataType::FLOAT32));
    cnn_cache_ =
        std::move(paddle::zeros({0, 0, 0, 0}, paddle::DataType::FLOAT32));

    encoder_outs_.clear();
    VLOG(3) << "u2nnet reset";
}

// Debug API
void U2Nnet::FeedEncoderOuts(const paddle::Tensor& encoder_out) {
    // encoder_out (T,D)
    encoder_outs_.clear();
    encoder_outs_.push_back(encoder_out);
}


void U2Nnet::FeedForward(const std::vector<BaseFloat>& features,
                         const int32& feature_dim,
                         NnetOut* out) {
    kaldi::Timer timer;

    std::vector<kaldi::BaseFloat> ctc_probs;
    ForwardEncoderChunkImpl(
        features, feature_dim, &out->logprobs, &out->vocab_dim);
    VLOG(1) << "FeedForward cost: " << timer.Elapsed() << " sec. "
            << features.size() / feature_dim << " frames.";
}


void U2Nnet::ForwardEncoderChunkImpl(
    const std::vector<kaldi::BaseFloat>& chunk_feats,
    const int32& feat_dim,
    std::vector<kaldi::BaseFloat>* out_prob,
    int32* vocab_dim) {
#ifdef WITH_PROFILING
    RecordEvent event(
        "ForwardEncoderChunkImpl", TracerEventType::UserDefined, 1);
#endif

    // 1. splice cached_feature, and chunk_feats
    //  First dimension is B, which is 1.
    // int num_frames = cached_feats_.size() + chunk_feats.size();

    int num_frames = chunk_feats.size() / feat_dim;
    VLOG(3) << "num_frames: " << num_frames;
    VLOG(3) << "feat_dim: " << feat_dim;

    // feats (B=1,T,D)
    paddle::Tensor feats =
        paddle::zeros({1, num_frames, feat_dim}, paddle::DataType::FLOAT32);
    float* feats_ptr = feats.mutable_data<float>();

    // not cache feature in nnet
    CHECK_EQ(cached_feats_.size(), 0);
    // CHECK_EQ(std::is_same<float, kaldi::BaseFloat>::value, true);
    std::memcpy(feats_ptr,
                chunk_feats.data(),
                chunk_feats.size() * sizeof(kaldi::BaseFloat));

    VLOG(3) << "feats shape: " << feats.shape()[0] << ", " << feats.shape()[1]
            << ", " << feats.shape()[2];

#ifndef NDEBUG
    {
        std::stringstream path("feat", std::ios_base::app | std::ios_base::out);
        path << offset_;
        std::ofstream feat_fobj(path.str().c_str(), std::ios::out);
        CHECK(feat_fobj.is_open());
        // feat_fobj << feats.shape()[0] << " " << feats.shape()[1] << " "
        //           << feats.shape()[2] << "\n";
        for (int i = 0; i < feats.numel(); i++) {
            feat_fobj << std::setprecision(18) << feats_ptr[i] << " ";
            if ((i + 1) % feat_dim == 0) {
                feat_fobj << "\n";
            }
        }
        feat_fobj << "\n";
    }
#endif

// Endocer chunk forward
#ifdef WITH_GPU
    feats = feats.copy_to(paddle::GPUPlace(), /*blocking*/ false);
    att_cache_ = att_cache_.copy_to(paddle::GPUPlace()), /*blocking*/ false;
    cnn_cache_ = cnn_cache_.copy_to(Paddle::GPUPlace(), /*blocking*/ false);
#endif

    int required_cache_size = num_left_chunks_ * chunk_size_;  // -1 * 16
    // must be scalar, but paddle do not have scalar.
    paddle::Tensor offset = paddle::full({1}, offset_, paddle::DataType::INT32);
    // freeze `required_cache_size` in graph, so not specific it in function
    // call.
    std::vector<paddle::Tensor> inputs = {
        feats, offset, /*required_cache_size, */ att_cache_, cnn_cache_};
    CHECK_EQ(inputs.size(), 4);
    std::vector<paddle::Tensor> outputs = forward_encoder_chunk_(inputs);
    CHECK_EQ(outputs.size(), 3);

#ifdef WITH_GPU
    paddle::Tensor chunk_out = outputs[0].copy_to(paddle::CPUPlace());
    att_cache_ = outputs[1].copy_to(paddle::CPUPlace());
    cnn_cache_ = outputs[2].copy_to(paddle::CPUPlace());
#else
    paddle::Tensor chunk_out = outputs[0];
    att_cache_ = outputs[1];
    cnn_cache_ = outputs[2];
#endif

#ifndef NDEBUG
    {
        std::stringstream path("encoder_logits",
                               std::ios_base::app | std::ios_base::out);
        auto i = offset_ - chunk_out.shape()[1];
        path << std::max(i, 0L);
        std::ofstream logits_fobj(path.str().c_str(), std::ios::out);
        CHECK(logits_fobj.is_open());
        logits_fobj << chunk_out.shape()[0] << " " << chunk_out.shape()[1]
                    << " " << chunk_out.shape()[2] << "\n";
        const float* chunk_out_ptr = chunk_out.data<float>();
        logits_fobj << chunk_out_ptr << std::endl;
        for (int i = 0; i < chunk_out.numel(); i++) {
            logits_fobj << chunk_out_ptr[i] << " ";
        }
        logits_fobj << "\n";
    }
#endif  // end TEST_DEBUG

    // current offset in decoder frame
    // not used in nnet
    offset_ += chunk_out.shape()[1];
    VLOG(2) << "encoder out chunk size: " << chunk_out.shape()[1]
            << " total: " << offset_;


    // collects encoder outs.
    encoder_outs_.push_back(chunk_out);
    VLOG(2) << "encoder_outs_ size: " << encoder_outs_.size();

#ifndef NDEBUG
    {
        std::stringstream path("encoder_logits_list",
                               std::ios_base::app | std::ios_base::out);
        path << offset_ - encoder_outs_[0].shape()[1];
        std::ofstream logits_out_fobj(path.str().c_str(), std::ios::out);
        CHECK(logits_out_fobj.is_open());
        logits_out_fobj << encoder_outs_[0].shape()[0] << " "
                        << encoder_outs_[0].shape()[1] << " "
                        << encoder_outs_[0].shape()[2] << "\n";
        const float* encoder_outs_ptr = encoder_outs_[0].data<float>();
        logits_out_fobj << encoder_outs_ptr << std::endl;
        for (int i = 0; i < encoder_outs_[0].numel(); i++) {
            logits_out_fobj << encoder_outs_ptr[i] << " ";
        }
        logits_out_fobj << "\n";
    }
#endif  // end TEST_DEBUG

#ifdef WITH_GPU

#error "Not implementation."

#else
    // compute ctc_activation == log_softmax
    inputs.clear();
    outputs.clear();
    inputs.push_back(chunk_out);
    CHECK_EQ(inputs.size(), 1);
    outputs = ctc_activation_(inputs);
    CHECK_EQ(outputs.size(), 1);
    paddle::Tensor ctc_log_probs = outputs[0];

#ifndef NDEBUG
    {
        std::stringstream path("encoder_logprob",
                               std::ios_base::app | std::ios_base::out);
        path << offset_ - chunk_out.shape()[1];

        std::ofstream logprob_fobj(path.str().c_str(), std::ios::out);
        CHECK(logprob_fobj.is_open());
        logprob_fobj << ctc_log_probs.shape()[0] << " "
                     << ctc_log_probs.shape()[1] << " "
                     << ctc_log_probs.shape()[2] << "\n";
        const float* logprob_ptr = ctc_log_probs.data<float>();
        for (int i = 0; i < ctc_log_probs.numel(); i++) {
            logprob_fobj << logprob_ptr[i] << " ";
            if ((i + 1) % ctc_log_probs.shape()[2] == 0) {
                logprob_fobj << "\n";
            }
        }
        logprob_fobj << "\n";
    }
#endif  // end TEST_DEBUG

#endif  // end WITH_GPU

    // Copy to output, (B=1,T,D)
    std::vector<int64_t> ctc_log_probs_shape = ctc_log_probs.shape();
    CHECK_EQ(ctc_log_probs_shape.size(), 3);
    int B = ctc_log_probs_shape[0];
    CHECK_EQ(B, 1);
    int T = ctc_log_probs_shape[1];
    int D = ctc_log_probs_shape[2];
    *vocab_dim = D;

    float* ctc_log_probs_ptr = ctc_log_probs.data<float>();

    out_prob->resize(T * D);
    std::memcpy(
        out_prob->data(), ctc_log_probs_ptr, T * D * sizeof(kaldi::BaseFloat));

#ifndef NDEBUG
    {
        std::stringstream path("encoder_logits_list_ctc",
                               std::ios_base::app | std::ios_base::out);
        path << offset_ - encoder_outs_[0].shape()[1];
        std::ofstream logits_out_fobj(path.str().c_str(), std::ios::out);
        CHECK(logits_out_fobj.is_open());
        logits_out_fobj << encoder_outs_[0].shape()[0] << " "
                        << encoder_outs_[0].shape()[1] << " "
                        << encoder_outs_[0].shape()[2] << "\n";
        const float* encoder_outs_ptr = encoder_outs_[0].data<float>();
        logits_out_fobj << encoder_outs_ptr << std::endl;
        for (int i = 0; i < encoder_outs_[0].numel(); i++) {
            logits_out_fobj << encoder_outs_ptr[i] << " ";
        }
        logits_out_fobj << "\n";
    }
#endif  // end TEST_DEBUG

    return;
}

float U2Nnet::ComputePathScore(const paddle::Tensor& prob,
                               const std::vector<int>& hyp,
                               int eos) {
    // sum `hyp` path scores in `prob`
    // prob (1, Umax, V)
    // hyp (U,)
    float score = 0.0f;
    std::vector<int64_t> dims = prob.shape();
    CHECK_EQ(dims.size(), 3);
    VLOG(2) << "prob shape: " << dims[0] << ", " << dims[1] << ", " << dims[2];
    CHECK_EQ(dims[0], 1);
    int vocab_dim = static_cast<int>(dims[2]);

    const float* prob_ptr = prob.data<float>();
    for (size_t i = 0; i < hyp.size(); ++i) {
        const float* row = prob_ptr + i * vocab_dim;
        score += row[hyp[i]];
    }
    const float* row = prob_ptr + hyp.size() * vocab_dim;
    score += row[eos];
    return score;
}


void U2Nnet::AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                                float reverse_weight,
                                std::vector<float>* rescoring_score) {
#ifdef WITH_PROFILING
    RecordEvent event("AttentionRescoring", TracerEventType::UserDefined, 1);
#endif
    CHECK(rescoring_score != nullptr);

    int num_hyps = hyps.size();
    rescoring_score->resize(num_hyps, 0.0f);

    if (num_hyps == 0) return;
    VLOG(2) << "num hyps: " << num_hyps;

    if (encoder_outs_.size() == 0) {
        // no encoder outs
        std::cerr << "encoder_outs_.size() is zero. Please check it."
                  << std::endl;
        return;
    }

    // prepare input
    paddle::Tensor hyps_lens =
        paddle::zeros({num_hyps}, paddle::DataType::INT64);
    int64_t* hyps_len_ptr = hyps_lens.mutable_data<int64_t>();
    int max_hyps_len = 0;
    for (size_t i = 0; i < num_hyps; ++i) {
        int len = hyps[i].size() + 1;  // eos
        max_hyps_len = std::max(max_hyps_len, len);
        hyps_len_ptr[i] = static_cast<int64_t>(len);
    }
    VLOG(2) << "max_hyps_len: " << max_hyps_len;

    paddle::Tensor hyps_tensor =
        paddle::full({num_hyps, max_hyps_len}, eos_, paddle::DataType::INT64);
    int64_t* hyps_ptr = hyps_tensor.mutable_data<int64_t>();
    for (size_t i = 0; i < num_hyps; ++i) {
        const std::vector<int>& hyp = hyps[i];
        int64_t* row = hyps_ptr + max_hyps_len * i;
        row[0] = sos_;
        for (size_t j = 0; j < hyp.size(); ++j) {
            row[j + 1] = hyp[j];
        }
    }

#ifndef NDEBUG
    {
        std::stringstream path("encoder_logits_concat",
                               std::ios_base::app | std::ios_base::out);
        for (int j = 0; j < encoder_outs_.size(); j++) {
            path << j;
            std::ofstream logits_out_fobj(path.str().c_str(), std::ios::out);
            CHECK(logits_out_fobj.is_open());
            logits_out_fobj << encoder_outs_[j].shape()[0] << " "
                            << encoder_outs_[j].shape()[1] << " "
                            << encoder_outs_[j].shape()[2] << "\n";
            const float* encoder_outs_ptr = encoder_outs_[j].data<float>();
            for (int i = 0; i < encoder_outs_[j].numel(); i++) {
                logits_out_fobj << encoder_outs_ptr[i] << " ";
            }
            logits_out_fobj << "\n";
        }
    }
#endif  // end TEST_DEBUG

    // forward attention decoder by hyps and correspoinding encoder_outs_
    paddle::Tensor encoder_out = paddle::concat(encoder_outs_, 1);
    VLOG(2) << "encoder_outs_ size: " << encoder_outs_.size();

#ifndef NDEBUG
    {
        std::stringstream path("encoder_out0",
                               std::ios_base::app | std::ios_base::out);
        std::ofstream encoder_out_fobj(path.str().c_str(), std::ios::out);
        CHECK(encoder_out_fobj.is_open());

        encoder_out_fobj << encoder_outs_[0].shape()[0] << " "
                         << encoder_outs_[0].shape()[1] << " "
                         << encoder_outs_[0].shape()[2] << "\n";
        const float* enc_logprob_ptr = encoder_outs_[0].data<float>();

        size_t size = encoder_outs_[0].numel();
        for (int i = 0; i < size; i++) {
            encoder_out_fobj << enc_logprob_ptr[i] << "\n";
        }
    }
#endif  // end TEST_DEBUG

#ifndef NDEBUG
    {
        std::stringstream path("encoder_out",
                               std::ios_base::app | std::ios_base::out);
        std::ofstream encoder_out_fobj(path.str().c_str(), std::ios::out);
        CHECK(encoder_out_fobj.is_open());

        encoder_out_fobj << encoder_out.shape()[0] << " "
                         << encoder_out.shape()[1] << " "
                         << encoder_out.shape()[2] << "\n";
        const float* enc_logprob_ptr = encoder_out.data<float>();

        size_t size = encoder_out.numel();
        for (int i = 0; i < size; i++) {
            encoder_out_fobj << enc_logprob_ptr[i] << "\n";
        }
    }
#endif  // end TEST_DEBUG

    std::vector<paddle::Tensor> inputs{
        hyps_tensor, hyps_lens, encoder_out};
    std::vector<paddle::Tensor> outputs = forward_attention_decoder_(inputs);
    CHECK_EQ(outputs.size(), 2);

    // (B, Umax, V)
    paddle::Tensor probs = outputs[0];
    std::vector<int64_t> probs_shape = probs.shape();
    CHECK_EQ(probs_shape.size(), 3);
    CHECK_EQ(probs_shape[0], num_hyps);
    CHECK_EQ(probs_shape[1], max_hyps_len);

#ifndef NDEBUG
    {
        std::stringstream path("decoder_logprob",
                               std::ios_base::app | std::ios_base::out);
        std::ofstream dec_logprob_fobj(path.str().c_str(), std::ios::out);
        CHECK(dec_logprob_fobj.is_open());

        dec_logprob_fobj << probs.shape()[0] << " " << probs.shape()[1] << " "
                         << probs.shape()[2] << "\n";
        const float* dec_logprob_ptr = probs.data<float>();

        size_t size = probs.numel();
        for (int i = 0; i < size; i++) {
            dec_logprob_fobj << dec_logprob_ptr[i] << "\n";
        }
    }
#endif  // end TEST_DEBUG

#ifndef NDEBUG
    {
        std::stringstream path("hyps_lens",
                               std::ios_base::app | std::ios_base::out);
        std::ofstream hyps_len_fobj(path.str().c_str(), std::ios::out);
        CHECK(hyps_len_fobj.is_open());

        const int64_t* hyps_lens_ptr = hyps_lens.data<int64_t>();

        size_t size = hyps_lens.numel();
        for (int i = 0; i < size; i++) {
            hyps_len_fobj << hyps_lens_ptr[i] << "\n";
        }
    }
#endif  // end TEST_DEBUG

#ifndef NDEBUG
    {
        std::stringstream path("hyps_tensor",
                               std::ios_base::app | std::ios_base::out);
        std::ofstream hyps_tensor_fobj(path.str().c_str(), std::ios::out);
        CHECK(hyps_tensor_fobj.is_open());

        const int64_t* hyps_tensor_ptr = hyps_tensor.data<int64_t>();

        size_t size = hyps_tensor.numel();
        for (int i = 0; i < size; i++) {
            hyps_tensor_fobj << hyps_tensor_ptr[i] << "\n";
        }
    }
#endif  // end TEST_DEBUG

    paddle::Tensor r_probs = outputs[1];
    std::vector<int64_t> r_probs_shape = r_probs.shape();
    if (is_bidecoder_ && reverse_weight > 0) {
        CHECK_EQ(r_probs_shape.size(), 3);
        CHECK_EQ(r_probs_shape[0], num_hyps);
        CHECK_EQ(r_probs_shape[1], max_hyps_len);
    } else {
        // dump r_probs
        CHECK_EQ(r_probs_shape.size(), 1);
        //CHECK_EQ(r_probs_shape[0], 1) << r_probs_shape[0];
    }

    // compute rescoring score
    using IntArray = paddle::experimental::IntArray;
    std::vector<paddle::Tensor> probs_v =
        paddle::experimental::split_with_num(probs, num_hyps, 0);
    VLOG(2) << "split prob: " << probs_v.size() << " "
            << probs_v[0].shape().size() << " 0: " << probs_v[0].shape()[0]
            << ", " << probs_v[0].shape()[1] << ", " << probs_v[0].shape()[2];
    //CHECK(static_cast<int>(probs_v.size()) == num_hyps)
     //   << ": is " << probs_v.size() << " expect: " << num_hyps;

    std::vector<paddle::Tensor> r_probs_v;
    if (is_bidecoder_ && reverse_weight > 0) {
        r_probs_v = paddle::experimental::split_with_num(r_probs, num_hyps, 0);
        //CHECK(static_cast<int>(r_probs_v.size()) == num_hyps)
         //   << "r_probs_v size: is " << r_probs_v.size()
          //  << " expect: " << num_hyps;
    }

    for (int i = 0; i < num_hyps; ++i) {
        const std::vector<int>& hyp = hyps[i];

        // left-to-right decoder score
        float score = 0.0f;
        score = ComputePathScore(probs_v[i], hyp, eos_);

        // right-to-left decoder score
        float r_score = 0.0f;
        if (is_bidecoder_ && reverse_weight > 0) {
            std::vector<int> r_hyp(hyp.size());
            std::reverse_copy(hyp.begin(), hyp.end(), r_hyp.begin());
            r_score = ComputePathScore(r_probs_v[i], r_hyp, eos_);
        }

        // combinded left-to-right and right-to-lfet score
        (*rescoring_score)[i] =
            score * (1 - reverse_weight) + r_score * reverse_weight;
        VLOG(3) << "hyp " << i << " " << hyp.size() << " score: " << score
                << " r_score: " << r_score
                << " reverse_weight: " << reverse_weight
                << " final score: " << (*rescoring_score)[i];
    }
}


void U2Nnet::EncoderOuts(
    std::vector<std::vector<kaldi::BaseFloat>>* encoder_out) const {
    // list of (B=1,T,D)
    int size = encoder_outs_.size();
    VLOG(3) << "encoder_outs_ size: " << size;

    for (int i = 0; i < size; i++) {
        const paddle::Tensor& item = encoder_outs_[i];
        const std::vector<int64_t> shape = item.shape();
        CHECK_EQ(shape.size(), 3);
        const int& B = shape[0];
        const int& T = shape[1];
        const int& D = shape[2];
        //CHECK(B == 1) << "Only support batch one.";
        VLOG(3) << "encoder out " << i << " shape: (" << B << "," << T << ","
                << D << ")";

        const float* this_tensor_ptr = item.data<float>();
        for (int j = 0; j < T; j++) {
            const float* cur = this_tensor_ptr + j * D;
            std::vector<kaldi::BaseFloat> out(D);
            std::memcpy(out.data(), cur, D * sizeof(kaldi::BaseFloat));
            encoder_out->emplace_back(out);
        }
    }
}

}  // namespace ppspeech
