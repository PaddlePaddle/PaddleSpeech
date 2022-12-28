# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import os
import unittest
import urllib.request

mono_channel_wav = 'https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav'
multi_channels_wav = 'https://paddlespeech.bj.bcebos.com/PaddleAudio/cat.wav'


class BackendTest(unittest.TestCase):
    def setUp(self):
        self.initWavInput()

    def initWavInput(self):
        self.files = []
        for url in [mono_channel_wav, multi_channels_wav]:
            if not os.path.isfile(os.path.basename(url)):
                urllib.request.urlretrieve(url, os.path.basename(url))
            self.files.append(os.path.basename(url))

    def initParmas(self):
        raise NotImplementedError
