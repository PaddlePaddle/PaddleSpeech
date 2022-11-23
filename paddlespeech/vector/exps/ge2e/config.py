# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from yacs.config import CfgNode

_C = CfgNode()

data_config = _C.data = CfgNode()

## Audio volume normalization
data_config.audio_norm_target_dBFS = -30

## Audio sample rate
data_config.sampling_rate = 16000  # Hz

## Voice Activation Detection
# Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
# This sets the granularity of the VAD. Should not need to be changed.
data_config.vad_window_length = 30  # In milliseconds
# Number of frames to average together when performing the moving average smoothing.
# The larger this value, the larger the VAD variations must be to not get smoothed out.
data_config.vad_moving_average_width = 8
# Maximum number of consecutive silent frames a segment can have.
data_config.vad_max_silence_length = 6

## Mel-filterbank
data_config.mel_window_length = 25  # In milliseconds
data_config.mel_window_step = 10  # In milliseconds
data_config.n_mels = 40  # mel bands

# Number of spectrogram frames in a partial utterance
data_config.partial_n_frames = 160  # 1600 ms
data_config.min_pad_coverage = 0.75  # at least 75% of the audio is valid in a partial
data_config.partial_overlap_ratio = 0.5  # overlap ratio between ajancent partials

model_config = _C.model = CfgNode()
model_config.num_layers = 3
model_config.hidden_size = 256
model_config.embedding_size = 256  # output size

training_config = _C.training = CfgNode()
training_config.learning_rate_init = 1e-4
training_config.speakers_per_batch = 64
training_config.utterances_per_speaker = 10
training_config.max_iteration = 1560000
training_config.save_interval = 10000
training_config.valid_interval = 10000


def get_cfg_defaults():
    return _C.clone()
