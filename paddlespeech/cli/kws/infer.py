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
from collections import OrderedDict
from typing import List
from typing import Optional
from typing import Union

import paddle
import yaml
from paddleaudio import load
from paddleaudio.compliance.kaldi import fbank as kaldi_fbank

from ..executor import BaseExecutor
from ..log import logger
from ..utils import stats_wrapper

__all__ = ['KWSExecutor']


class KWSExecutor(BaseExecutor):
    def __init__(self):
        super().__init__(task='kws')
        self.parser = argparse.ArgumentParser(
            prog='paddlespeech.kws', add_help=True)
        self.parser.add_argument(
            '--input',
            type=str,
            default=None,
            help='Audio file to keyword spotting.')
        self.parser.add_argument(
            '--model',
            type=str,
            default='mdtc_heysnips',
            choices=[
                tag[:tag.index('-')]
                for tag in self.task_resource.pretrained_models.keys()
            ],
            help='Choose model type of kws task.')
        self.parser.add_argument(
            '--config',
            type=str,
            default=None,
            help='Config of kws task. Use deault config when it is None.')
        self.parser.add_argument(
            '--ckpt_path',
            type=str,
            default=None,
            help='Checkpoint file of model.')
        self.parser.add_argument(
            '--device',
            type=str,
            default=paddle.get_device(),
            help='Choose device to execute model inference.')
        self.parser.add_argument(
            '-d',
            '--job_dump_result',
            action='store_true',
            help='Save job result into file.')
        self.parser.add_argument(
            '-v',
            '--verbose',
            action='store_true',
            help='Increase logger verbosity of current task.')

    def _init_from_path(self,
                        model_type: str='mdtc_heysnips',
                        cfg_path: Optional[os.PathLike]=None,
                        ckpt_path: Optional[os.PathLike]=None):
        """
            Init model and other resources from a specific path.
        """
        if hasattr(self, 'model'):
            logger.info('Model had been initialized.')
            return

        if ckpt_path is None:
            tag = model_type + '-' + '16k'
            self.task_resource.set_task_model(tag)
            self.cfg_path = os.path.join(
                self.task_resource.res_dir,
                self.task_resource.res_dict['cfg_path'])
            self.ckpt_path = os.path.join(
                self.task_resource.res_dir,
                self.task_resource.res_dict['ckpt_path'])
        else:
            self.cfg_path = os.path.abspath(cfg_path)
            self.ckpt_path = os.path.abspath(ckpt_path)

        # config
        with open(self.cfg_path, 'r') as f:
            config = yaml.safe_load(f)

        # model
        model_class = self.task_resource.get_model_class(model_type)
        model_dict = paddle.load(self.ckpt_path)
        self.model = model_class(config['model'])
        self.feature_extractor = kaldi_fbank(config['feat'])
        self.model.set_state_dict(model_dict)
        self.model.eval()

    def preprocess(self, audio_file: Union[str, os.PathLike]):
        """
            Input preprocess and return paddle.Tensor stored in self.input.
            Input content can be a text(tts), a file(asr, cls) or a streaming(not supported yet).
        """
        waveform, _ = load(audio_file)
        if isinstance(audio_file, (str, os.PathLike)):
            logger.info("Preprocessing audio_file:" + audio_file)

        # Feature extraction
        self._inputs['feats'] = self.feature_extractor(waveform)

    @paddle.no_grad()
    def infer(self):
        """
            Model inference and result stored in self.output.
        """
        self._outputs['logits'] = self.model(self._inputs['feats'])

    def postprocess(self) -> Union[str, os.PathLike]:
        """
            Output postprocess and return human-readable results such as texts and audio files.
        """
        return max(self._outputs['logits'][0])

    def execute(self, argv: List[str]) -> bool:
        """
            Command line entry.
        """
        parser_args = self.parser.parse_args(argv)

        model_type = parser_args.model
        cfg_path = parser_args.config
        ckpt_path = parser_args.ckpt_path
        device = parser_args.device

        if not parser_args.verbose:
            self.disable_task_loggers()

        task_source = self.get_input_source(parser_args.input)
        task_results = OrderedDict()
        has_exceptions = False

        for id_, input_ in task_source.items():
            try:
                res = self(input_, model_type, cfg_path, ckpt_path, device)
                task_results[id_] = res
            except Exception as e:
                has_exceptions = True
                task_results[id_] = f'{e.__class__.__name__}: {e}'

        self.process_task_results(parser_args.input, task_results,
                                  parser_args.job_dump_result)

        if has_exceptions:
            return False
        else:
            return True

    @stats_wrapper
    def __call__(self,
                 audio_file: os.PathLike,
                 model: str='mdtc_heysnips',
                 config: Optional[os.PathLike]=None,
                 ckpt_path: Optional[os.PathLike]=None,
                 device: str=paddle.get_device()):
        """
            Python API to call an executor.
        """
        audio_file = os.path.abspath(os.path.expanduser(audio_file))
        paddle.set_device(device)
        self._init_from_path(model, config, ckpt_path)
        self.preprocess(audio_file)
        self.infer()
        res = self.postprocess()

        return res
