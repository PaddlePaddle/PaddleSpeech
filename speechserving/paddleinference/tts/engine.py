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
import argparse
import os
from typing import Any
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import paddle
import soundfile as sf
import yaml
from yacs.config import CfgNode

from paddlespeech.cli.log import logger
from paddlespeech.cli.utils import cli_register
from paddlespeech.cli.utils import download_and_decompress
from paddlespeech.cli.utils import MODEL_HOME

from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.t2s.frontend import English
from paddlespeech.t2s.frontend.zh_frontend import Frontend
from paddlespeech.t2s.modules.normalizer import ZScore

from ttsserverexecutor import TTSServerExecutor

from pattern_singleton import Singleton

class TTSEngine(metaclass=Singleton):
    """[summary]

    Args:
        metaclass ([type], optional): [description]. Defaults to Singleton.
    """
    def __init__(self):
        self.ttsexecutor = TTSServerExecutor()
        self.conf_dict = self.get_conf()
        self.ttsexecutor._init_from_path(
            am=self.conf_dict["am"],
            am_config=self.conf_dict["am_config"],
            am_ckpt=self.conf_dict["am_ckpt"],
            am_stat=self.conf_dict["am_stat"],
            phones_dict=self.conf_dict["phones_dict"],
            tones_dict=self.conf_dict["tones_dict"],
            speaker_dict=self.conf_dict["speaker_dict"],
            voc=self.conf_dict["voc"],
            voc_config=self.conf_dict["voc_config"],
            voc_ckpt=self.conf_dict["voc_ckpt"],
            voc_stat=self.conf_dict["voc_stat"],
            lang=self.conf_dict["lang"])

    def get_conf(self):

        config_path = self.ttsexecutor.parser.parse_args().conf
        with open(config_path, 'rt') as f:
            config = CfgNode(yaml.safe_load(f))

        conf_dict = {}
        conf_dict["am"] = config.am
        conf_dict["am_config"] = config.am_config
        conf_dict["am_ckpt"] = config.am_ckpt
        conf_dict["am_stat"] = config.am_stat
        conf_dict["phones_dict"] = config.phones_dict
        conf_dict["tones_dict"] = config.tones_dict
        conf_dict["speaker_dict"] = config.speaker_dict
        #conf_dict["spk_id"] = config.spk_id
        conf_dict["voc"] = config.voc
        conf_dict["voc_config"] = config.voc_config
        conf_dict["voc_ckpt"] = config.voc_ckpt
        conf_dict["voc_stat"] = config.voc_stat
        conf_dict["lang"] = config.lang
        conf_dict["device"] = config.device
        
        print("Get configuration parameters successfully")
        return conf_dict

    def postprocess(self, 
                    wav, 
                    original_fs: int, 
                    target_fs: int=0, 
                    volume: float=1.0, 
                    speed: float=1.0, 
                    audio_path: str=None, 
                    audio_format: str="wav"):
        """Post-processing operations, including speech, volume, sample rate, save audio file

        Args:
            wav (numpy(float)): Synthesized audio sample points
            original_fs (int): original audio sample rate
            target_fs (int): target audio sample rate
            volume (float): target volume
            speed (float): target speed
        """
        # transform sample_rate
        if target_fs == 0:
            target_fs = original_fs
        #TODO

        # transform volume
        # TODO

        # transform speed
        # TODO
        target_wav = wav

        # save audio
        if audio_path is not None:
            pass
            # TODO

        return target_wav, target_fs
    
    def run(self, 
            sentence: str, 
            spk_id: int=0, 
            speed: float=1.0, 
            volume: float=1.0, 
            sample_rate: int=0, 
            tts_audio_path: str=None, 
            audio_format: str="wav"):

        self.ttsexecutor.infer(
            text=sentence, 
            lang=self.conf_dict["lang"], 
            am=self.conf_dict["am"], 
            spk_id=spk_id)

        target_wav, target_fs = self.postprocess(
            wav=self.ttsexecutor._outputs['wav'].numpy(), 
            original_fs=self.ttsexecutor.am_config.fs,
            target_fs=sample_rate,
            volume=volume,
            speed=speed,
            audio_path=tts_audio_path,
            audio_format=audio_format)
        
        return target_wav, target_fs
