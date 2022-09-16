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

from ..executor import BaseExecutor
from ..log import logger
from ..utils import stats_wrapper
from paddleaudio.backends import soundfile_load as load_audio
from paddleaudio.compliance.kaldi import fbank as kaldi_fbank

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
            '--threshold',
            type=float,
            default=0.8,
            help='Score threshold for keyword spotting.')
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
            logger.debug('Model had been initialized.')
            return

        if ckpt_path is None:
            tag = model_type + '-' + '16k'
            self.task_resource.set_task_model(tag)
            self.cfg_path = os.path.join(
                self.task_resource.res_dir,
                self.task_resource.res_dict['cfg_path'])
            self.ckpt_path = os.path.join(
                self.task_resource.res_dir,
                self.task_resource.res_dict['ckpt_path'] + '.pdparams')
        else:
            self.cfg_path = os.path.abspath(cfg_path)
            self.ckpt_path = os.path.abspath(ckpt_path)

        # config
        with open(self.cfg_path, 'r') as f:
            config = yaml.safe_load(f)

        # model
        backbone_class = self.task_resource.get_model_class(
            model_type.split('_')[0])
        model_class = self.task_resource.get_model_class(
            model_type.split('_')[0] + '_for_kws')
        backbone = backbone_class(
            stack_num=config['stack_num'],
            stack_size=config['stack_size'],
            in_channels=config['in_channels'],
            res_channels=config['res_channels'],
            kernel_size=config['kernel_size'],
            causal=True, )
        self.model = model_class(
            backbone=backbone, num_keywords=config['num_keywords'])
        model_dict = paddle.load(self.ckpt_path)
        self.model.set_state_dict(model_dict)
        self.model.eval()

        self.feature_extractor = lambda x: kaldi_fbank(
            x, sr=config['sample_rate'],
            frame_shift=config['frame_shift'],
            frame_length=config['frame_length'],
            n_mels=config['n_mels']
        )

    def preprocess(self, audio_file: Union[str, os.PathLike]):
        """
            Input preprocess and return paddle.Tensor stored in self.input.
            Input content can be a text(tts), a file(asr, cls) or a streaming(not supported yet).
        """
        assert os.path.isfile(audio_file)
        waveform, _ = load(audio_file)
        if isinstance(audio_file, (str, os.PathLike)):
            logger.debug("Preprocessing audio_file:" + audio_file)

        # Feature extraction
        waveform = paddle.to_tensor(waveform).unsqueeze(0)
        self._inputs['feats'] = self.feature_extractor(waveform).unsqueeze(0)

    @paddle.no_grad()
    def infer(self):
        """
            Model inference and result stored in self.output.
        """
        self._outputs['logits'] = self.model(self._inputs['feats'])

    def postprocess(self, threshold: float) -> Union[str, os.PathLike]:
        """
            Output postprocess and return human-readable results such as texts and audio files.
        """
        kws_score = max(self._outputs['logits'][0, :, 0]).item()
        return 'Score: {:.3f}, Threshold: {}, Is keyword: {}'.format(
            kws_score, threshold, kws_score > threshold)

    def execute(self, argv: List[str]) -> bool:
        """
            Command line entry.
        """
        parser_args = self.parser.parse_args(argv)

        model_type = parser_args.model
        cfg_path = parser_args.config
        ckpt_path = parser_args.ckpt_path
        device = parser_args.device
        threshold = parser_args.threshold

        if not parser_args.verbose:
            self.disable_task_loggers()

        task_source = self.get_input_source(parser_args.input)
        task_results = OrderedDict()
        has_exceptions = False

        for id_, input_ in task_source.items():
            try:
                res = self(input_, threshold, model_type, cfg_path, ckpt_path,
                           device)
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
                 threshold: float=0.8,
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
        res = self.postprocess(threshold)

        return res
