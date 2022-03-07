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

#include "frontend/linear_spectrogram.h"
#include "kaldi/base/kaldi-math.h"
#include "kaldi/matrix/matrix-functions.h"

namespace ppspeech {

using kaldi::int32;
using kaldi::BaseFloat;
using kaldi::Vector;
using kaldi::VectorBase;
using kaldi::Matrix;
using std::vector;

// todo remove later
void CopyVector2StdVector_(const VectorBase<BaseFloat>& input,
                           vector<BaseFloat>* output) {
    if (input.Dim() == 0) return;
    output->resize(input.Dim());
    for (size_t idx = 0; idx < input.Dim(); ++idx) {
        (*output)[idx] = input(idx);
    }
}

void CopyStdVector2Vector_(const vector<BaseFloat>& input,
                           Vector<BaseFloat>* output) {
    if (input.empty()) return;
    output->Resize(input.size());
    for (size_t idx = 0; idx < input.size(); ++idx) {
        (*output)(idx) = input[idx];
    }
}

LinearSpectrogram::LinearSpectrogram(
    const LinearSpectrogramOptions& opts,
    std::unique_ptr<FeatureExtractorInterface> base_extractor) {
    opts_ = opts;
    base_extractor_ = std::move(base_extractor);
    int32 window_size = opts.frame_opts.WindowSize();
    int32 window_shift = opts.frame_opts.WindowShift();
    fft_points_ = window_size;
    hanning_window_.resize(window_size);

    double a = M_2PI / (window_size - 1);
    hanning_window_energy_ = 0;
    for (int i = 0; i < window_size; ++i) {
        hanning_window_[i] = 0.5 - 0.5 * cos(a * i);
        hanning_window_energy_ += hanning_window_[i] * hanning_window_[i];
    }

    dim_ = fft_points_ / 2 + 1;  // the dimension is Fs/2 Hz
}

void LinearSpectrogram::AcceptWaveform(const VectorBase<BaseFloat>& input) {
    base_extractor_->AcceptWaveform(input);
}

void LinearSpectrogram::Hanning(vector<float>* data) const {
    CHECK_GE(data->size(), hanning_window_.size());

    for (size_t i = 0; i < hanning_window_.size(); ++i) {
        data->at(i) *= hanning_window_[i];
    }
}

bool LinearSpectrogram::NumpyFft(vector<BaseFloat>* v,
                                 vector<BaseFloat>* real,
                                 vector<BaseFloat>* img) const {
    Vector<BaseFloat> v_tmp;
    CopyStdVector2Vector_(*v, &v_tmp);
    RealFft(&v_tmp, true);
    CopyVector2StdVector_(v_tmp, v);
    real->push_back(v->at(0));
    img->push_back(0);
    for (int i = 1; i < v->size() / 2; i++) {
        real->push_back(v->at(2 * i));
        img->push_back(v->at(2 * i + 1));
    }
    real->push_back(v->at(1));
    img->push_back(0);

    return true;
}

// todo remove later
void LinearSpectrogram::ReadFeats(Matrix<BaseFloat>* feats) {
    Vector<BaseFloat> tmp;
    waveform_.Resize(base_extractor_->Dim());
    Compute(tmp, &waveform_);
    vector<vector<BaseFloat>> result;
    vector<BaseFloat> feats_vec;
    CopyVector2StdVector_(waveform_, &feats_vec);
    Compute(feats_vec, result);
    feats->Resize(result.size(), result[0].size());
    for (int row_idx = 0; row_idx < result.size(); ++row_idx) {
        for (int col_idx = 0; col_idx < result[0].size(); ++col_idx) {
            (*feats)(row_idx, col_idx) = result[row_idx][col_idx];
        }
    }
    waveform_.Resize(0);
}

void LinearSpectrogram::Read(VectorBase<BaseFloat>* feat) {
    // todo
    return;
}

// only for test, remove later
// todo: compute the feature frame by frame.
void LinearSpectrogram::Compute(const VectorBase<kaldi::BaseFloat>& input,
                                VectorBase<kaldi::BaseFloat>* feature) {
    base_extractor_->Read(feature);
}

// Compute spectrogram feat, only for test, remove later
// todo: refactor later (SmileGoat)
bool LinearSpectrogram::Compute(const vector<float>& wave,
                                vector<vector<float>>& feat) {
    int num_samples = wave.size();
    const int& frame_length = opts_.frame_opts.WindowSize();
    const int& sample_rate = opts_.frame_opts.samp_freq;
    const int& frame_shift = opts_.frame_opts.WindowShift();
    const int& fft_points = fft_points_;
    const float scale = hanning_window_energy_ * sample_rate;

    if (num_samples < frame_length) {
        return true;
    }

    int num_frames = 1 + ((num_samples - frame_length) / frame_shift);
    feat.resize(num_frames);
    vector<float> fft_real((fft_points_ / 2 + 1), 0);
    vector<float> fft_img((fft_points_ / 2 + 1), 0);
    vector<float> v(frame_length, 0);
    vector<float> power((fft_points / 2 + 1));

    for (int i = 0; i < num_frames; ++i) {
        vector<float> data(wave.data() + i * frame_shift,
                           wave.data() + i * frame_shift + frame_length);
        Hanning(&data);
        fft_img.clear();
        fft_real.clear();
        v.assign(data.begin(), data.end());
        NumpyFft(&v, &fft_real, &fft_img);

        feat[i].resize(fft_points / 2 + 1);  // the last dimension is Fs/2 Hz
        for (int j = 0; j < (fft_points / 2 + 1); ++j) {
            power[j] = fft_real[j] * fft_real[j] + fft_img[j] * fft_img[j];
            feat[i][j] = power[j];

            if (j == 0 || j == feat[0].size() - 1) {
                feat[i][j] /= scale;
            } else {
                feat[i][j] *= (2.0 / scale);
            }

            // log added eps=1e-14
            feat[i][j] = std::log(feat[i][j] + 1e-14);
        }
    }
    return true;
}

}  // namespace ppspeech