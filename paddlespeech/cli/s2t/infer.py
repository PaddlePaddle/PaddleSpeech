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
from typing import List
from typing import Optional
from typing import Union

import paddle

from ..executor import BaseExecutor
from ..utils import cli_register

__all__ = ['S2TExecutor']


@cli_register(
    name='paddlespeech.s2t', description='Speech to text infer command.')
class S2TExecutor(BaseExecutor):
    def __init__(self):
        super(S2TExecutor, self).__init__()

        self.parser = argparse.ArgumentParser(
            prog='paddlespeech.s2t', add_help=True)
        self.parser.add_argument(
            '--config',
            type=str,
            default=None,
            help='Config of s2t task. Use deault config when it is None.')
        self.parser.add_argument(
            '--input', type=str, help='Audio file to recognize.')
        self.parser.add_argument(
            '--device',
            type=str,
            default='cpu',
            help='Choose device to execute model inference.')

    def _get_default_cfg_path(self):
        """
            Returns a default config file path of current task.
        """
        pass

    def _init_from_cfg(self, cfg_path: Optional[os.PathLike]=None):
        """
            Init model from a specific config file.
        """
        pass

    def preprocess(self, input: Union[str, os.PathLike]):
        """
            Input preprocess and return paddle.Tensor stored in self.input.
            Input content can be a text(t2s), a file(s2t, cls) or a streaming(not supported yet).
        """
        pass

    @paddle.no_grad()
    def infer(self):
        """
            Model inference and result stored in self.output.
        """
        pass

    def postprocess(self) -> Union[str, os.PathLike]:
        """
            Output postprocess and return human-readable results such as texts and audio files.
        """
        pass

    def execute(self, argv: List[str]) -> bool:
        parser_args = self.parser.parse_args(argv)
        print(parser_args)

        config = parser_args.config
        audio_file = parser_args.input
        device = parser_args.device

        if config is not None:
            assert os.path.isfile(config), 'Config file is not valid.'
        else:
            config = self._get_default_cfg_path()

        try:
            self._init_from_cfg(config)
            self.preprocess(audio_file)
            self.infer()
            res = self.postprocess()  # Retrieve result of s2t.
            print(res)
            return True
        except Exception as e:
            print(e)
            return False
