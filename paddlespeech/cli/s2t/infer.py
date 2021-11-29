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
from ..utils import download_and_decompress
from ..utils import logger
from ..utils import MODEL_HOME

__all__ = ['S2TExecutor']

pretrained_models = {
    "wenetspeech_zh": {
        'url':
        'https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/conformer.model.tar.gz',
        'md5':
        '54e7a558a6e020c2f5fb224874943f97',
    }
}


@cli_register(
    name='paddlespeech.s2t', description='Speech to text infer command.')
class S2TExecutor(BaseExecutor):
    def __init__(self):
        super(S2TExecutor, self).__init__()

        self.parser = argparse.ArgumentParser(
            prog='paddlespeech.s2t', add_help=True)
        self.parser.add_argument(
            '--model',
            type=str,
            default='wenetspeech',
            help='Choose model type of asr task.')
        self.parser.add_argument(
            '--lang', type=str, default='zh', help='Choose model language.')
        self.parser.add_argument(
            '--config',
            type=str,
            default=None,
            help='Config of s2t task. Use deault config when it is None.')
        self.parser.add_argument(
            '--ckpt_path',
            type=str,
            default=None,
            help='Checkpoint file of model.')
        self.parser.add_argument(
            '--input', type=str, help='Audio file to recognize.')
        self.parser.add_argument(
            '--device',
            type=str,
            default='cpu',
            help='Choose device to execute model inference.')

    def _get_pretrained_path(self, tag: str) -> os.PathLike:
        """
            Download and returns pretrained resources path of current task.
        """
        assert tag in pretrained_models, 'Can not find pretrained resources of {}.'.format(
            tag)

        res_path = os.path.join(MODEL_HOME, tag)
        decompressed_path = download_and_decompress(pretrained_models[tag],
                                                    res_path)
        logger.info(
            'Use pretrained model stored in: {}'.format(decompressed_path))
        return decompressed_path

    def _init_from_path(self,
                        model_type: str='wenetspeech',
                        lang: str='zh',
                        cfg_path: Optional[os.PathLike]=None,
                        ckpt_path: Optional[os.PathLike]=None):
        """
            Init model and other resources from a specific path.
        """
        if cfg_path is None or ckpt_path is None:
            res_path = self._get_pretrained_path(
                model_type + '_' + lang)  # wenetspeech_zh
            cfg_path = os.path.join(res_path, 'conf/conformer.yaml')
            ckpt_path = os.path.join(
                res_path, 'exp/conformer/checkpoints/wenetspeech.pdparams')
            logger.info(res_path)
            logger.info(cfg_path)
            logger.info(ckpt_path)

        # Init body.
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

        model = parser_args.model
        lang = parser_args.lang
        config = parser_args.config
        ckpt_path = parser_args.ckpt_path
        audio_file = parser_args.input
        device = parser_args.device

        try:
            self._init_from_path(model, lang, config, ckpt_path)
            self.preprocess(audio_file)
            self.infer()
            res = self.postprocess()  # Retrieve result of s2t.
            print(res)
            return True
        except Exception as e:
            print(e)
            return False
