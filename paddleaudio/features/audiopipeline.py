#!/usr/bin/python3
#! coding:utf-8

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
import paddleaudio

from paddlespeech.s2t.frontend.augmentor.augmentation import AugmentationPipeline
from paddlespeech.s2t.frontend.featurizer.speech_featurizer import SpeechFeaturizer
from paddlespeech.s2t.frontend.speech import SpeechSegment
from paddlespeech.s2t.utils.log import Log
import json
import numpy as np

logger = Log(__name__).getlog()

class AudioPipeline:
    def __init__(self,
                 config,
                 train_dataset=True,
                 random_seed=0,
                 keep_transcription_text=True):
        # 数据增强
        self.augmentation = AugmentationPipeline(
            config.augmentation_config, random_seed=random_seed)

        # 提取特征
        self._speech_featurizer = SpeechFeaturizer(
            unit_type=config.unit_type,
            vocab_filepath=config.vocab_filepath,
            spm_model_prefix=config.spm_model_prefix,
            spectrum_type=config.spectrum_type,
            feat_dim=config.feat_dim,
            delta_delta=config.delta_delta,
            stride_ms=config.stride_ms,
            window_ms=config.window_ms,
            n_fft=config.n_fft,
            max_freq=config.max_freq,
            target_sample_rate=config.target_sample_rate,
            use_dB_normalization=config.use_dB_normalization,
            target_dB=config.target_dB,
            dither=config.dither)

        self.config = config
        self.keep_transcription_text = keep_transcription_text
        logger.info("audio pipeline construct successfully")

    def process_utterance(self, utt):
        wav_path = utt["wav"]
        start = float(utt["start"])
        end = float(utt["end"])
        sig, fs = paddleaudio.backends.audio.sound_file_load(
                        wav_path, offset=start, dtype="float32", duration=(end - start))
        speech_segment = SpeechSegment(sig, fs, "")
        
        # 对音频进行数据增强
        self.augmentation.transform_audio(speech_segment)
            
        # 提取音频特征
        spectrum, transcript_part = self._speech_featurizer.featurize(
                speech_segment, self.keep_transcription_text)
        
        return spectrum

    def __call__(self, data):
        eps = 1e-14

        for utt_id in data.keys():
            utt_data = data[utt_id]
            spectrum = self.process_utterance(utt_data)
            data[utt_id]["wav"] = spectrum # 将音频字段内容从采样点更新到特征数据
            
        return data
            # cmvn_std_filepath = self.config.mean_std_filepath
            # logger.info("process cmvn data: {}".format(cmvn_std_filepath))
            # if self.config.mean_std_filepath:
            #     all_mean_stat += np.sum(spectrum, axis=0)
            #     all_var_stat += np.sum(np.square(spectrum), axis=0)
            #     frame_num += spectrum.shape[0]

        # # for utt_id in data.keys():
        # #     logger.info("wav feature data: {}".format(data[utt_id]))
        # if self.config.mean_std_filepath and \
        #     frame_num != 0:
        #     # D[x] = 1/n E[X^{2}] - {1/n E[x]}^{2}
        #     all_mean_stat = all_mean_stat / (1.0 * frame_num)
        #     self.mean = np.clip(all_mean_stat, eps, np.finfo(np.float32).max)
            
        #     all_var_stat = all_var_stat / (1.0 * frame_num) - np.power(self.mean, 2)
        #     all_var_stat = np.clip(all_mean_stat, eps, np.finfo(np.float32).max)
        #     self.std = 1.0 / np.sqrt(all_var_stat)
            
        #     # 进行cmvn操作
        #     for utt_id in data.keys():
        #         data[utt_id]["wav"] = data[utt_id]["wav"] - self.mean
                
        #     # 保存 cmvn 数据
        #     cmvn_info = {
        #         "mean": list(self.mean.tolist()),
        #         "std" : list(self.std.tolist()),
        #         "frame_num": frame_num,
        #     }

        #     with open(self.config.mean_std_filepath, 'w') as f:
        #         json.dump(cmvn_info, f)


class CMVNNormalizer():
    def __init__(self, 
                 config,
                 data):
        self.config = config
        self.mean_std_filepath = config.mean_std_filepath
        self.mean = np.zeros(config.feat_dim)
        self.std = np.zeros(config.feat_dim)
        self.compute_cmvn(data)

    def compute_cmvn(self, data):
        all_mean_stat = np.zeros(self.config.feat_dim)
        all_var_stat = np.zeros(self.config.feat_dim)
        frame_num = 0
        eps = 1e-14
        for utt_id in data.keys():
            spectrum = data[utt_id]["wav"]
            all_mean_stat += np.sum(spectrum, axis=0)
            all_var_stat += np.sum(np.square(spectrum), axis=0)
            frame_num += spectrum.shape[0]
        
        # D[x] = 1/n E[X^{2}] - {1/n E[x]}^{2}
        all_mean_stat = all_mean_stat / (1.0 * frame_num)
        self.mean = np.clip(all_mean_stat, eps, np.finfo(np.float32).max)

        all_var_stat = all_var_stat / (1.0 * frame_num) - np.power(self.mean, 2)
        all_var_stat = np.clip(all_mean_stat, eps, np.finfo(np.float32).max)
        self.std = 1.0 / np.sqrt(all_var_stat)

        for utt_id in data.keys():
            data[utt_id]["wav"] = self.std * (data[utt_id]["wav"] - self.mean)

        # 保存 cmvn 数据
        cmvn_info = {
            "mean": list(self.mean.tolist()),
            "std" : list(self.std.tolist()),
            "frame_num": frame_num,
        } 
        with open(self.config.mean_std_filepath, 'w') as f:
            json.dump(cmvn_info, f)