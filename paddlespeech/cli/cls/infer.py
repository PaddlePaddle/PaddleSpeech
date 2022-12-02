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

import numpy as np
import paddle
import yaml
from paddle.audio.features import LogMelSpectrogram
from paddleaudio.backends import soundfile_load as load

from ..executor import BaseExecutor
from ..log import logger
from ..utils import stats_wrapper

__all__ = ['CLSExecutor']


class CLSExecutor(BaseExecutor):
    def __init__(self):
        super().__init__(task='cls')
        self.parser = argparse.ArgumentParser(
            prog='paddlespeech.cls', add_help=True)
        self.parser.add_argument(
            '--input', type=str, default=None, help='Audio file to classify.')
        self.parser.add_argument(
            '--model',
            type=str,
            default='panns_cnn14',
            choices=[
                tag[:tag.index('-')]
                for tag in self.task_resource.pretrained_models.keys()
            ],
            help='Choose model type of cls task.')
        self.parser.add_argument(
            '--config',
            type=str,
            default=None,
            help='Config of cls task. Use deault config when it is None.')
        self.parser.add_argument(
            '--ckpt_path',
            type=str,
            default=None,
            help='Checkpoint file of model.')
        self.parser.add_argument(
            '--label_file',
            type=str,
            default=None,
            help='Label file of cls task.')
        self.parser.add_argument(
            '--topk',
            type=int,
            default=1,
            help='Return topk scores of classification result.')
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
                        model_type: str='panns_cnn14',
                        cfg_path: Optional[os.PathLike]=None,
                        ckpt_path: Optional[os.PathLike]=None,
                        label_file: Optional[os.PathLike]=None):
        """
            Init model and other resources from a specific path.
        """
        if hasattr(self, 'model'):
            logger.debug('Model had been initialized.')
            return

        if label_file is None or ckpt_path is None:
            tag = model_type + '-' + '32k'  # panns_cnn14-32k
            self.task_resource.set_task_model(tag, version=None)
            self.cfg_path = os.path.join(
                self.task_resource.res_dir,
                self.task_resource.res_dict['cfg_path'])
            self.label_file = os.path.join(
                self.task_resource.res_dir,
                self.task_resource.res_dict['label_file'])
            self.ckpt_path = os.path.join(
                self.task_resource.res_dir,
                self.task_resource.res_dict['ckpt_path'])
        else:
            self.cfg_path = os.path.abspath(cfg_path)
            self.label_file = os.path.abspath(label_file)
            self.ckpt_path = os.path.abspath(ckpt_path)

        # config
        with open(self.cfg_path, 'r') as f:
            self._conf = yaml.safe_load(f)

        # labels
        self._label_list = []
        with open(self.label_file, 'r') as f:
            for line in f:
                self._label_list.append(line.strip())

        # model
        model_class = self.task_resource.get_model_class(model_type)
        model_dict = paddle.load(self.ckpt_path)
        self.model = model_class(extract_embedding=False)
        self.model.set_state_dict(model_dict)
        self.model.eval()

    def preprocess(self, audio_file: Union[str, os.PathLike]):
        """
            Input preprocess and return paddle.Tensor stored in self.input.
            Input content can be a text(tts), a file(asr, cls) or a streaming(not supported yet).
        """
        feat_conf = self._conf['feature']
        logger.debug(feat_conf)
        waveform, _ = load(
            file=audio_file,
            sr=feat_conf['sample_rate'],
            mono=True,
            dtype='float32')
        if isinstance(audio_file, (str, os.PathLike)):
            logger.debug("Preprocessing audio_file:" + audio_file)

        # Feature extraction
        feature_extractor = LogMelSpectrogram(
            sr=feat_conf['sample_rate'],
            n_fft=feat_conf['n_fft'],
            hop_length=feat_conf['hop_length'],
            window=feat_conf['window'],
            win_length=feat_conf['window_length'],
            f_min=feat_conf['f_min'],
            f_max=feat_conf['f_max'],
            n_mels=feat_conf['n_mels'], )
        feats = feature_extractor(
            paddle.to_tensor(paddle.to_tensor(waveform).unsqueeze(0)))
        self._inputs['feats'] = paddle.transpose(feats, [0, 2, 1]).unsqueeze(
            1)  # [B, N, T] -> [B, 1, T, N]

    @paddle.no_grad()
    def infer(self):
        """
            Model inference and result stored in self.output.
        """
        self._outputs['logits'] = self.model(self._inputs['feats'])

    def _generate_topk_label(self, result: np.ndarray, topk: int) -> str:
        assert topk <= len(
            self._label_list), 'Value of topk is larger than number of labels.'

        topk_idx = (-result).argsort()[:topk]
        ret = ''
        for idx in topk_idx:
            label, score = self._label_list[idx], result[idx]
            ret += f'{label} {score} '
        return ret

    def postprocess(self, topk: int) -> Union[str, os.PathLike]:
        """
            Output postprocess and return human-readable results such as texts and audio files.
        """
        return self._generate_topk_label(
            result=self._outputs['logits'].squeeze(0).numpy(), topk=topk)

    def execute(self, argv: List[str]) -> bool:
        """
            Command line entry.
        """
        parser_args = self.parser.parse_args(argv)

        model_type = parser_args.model
        label_file = parser_args.label_file
        cfg_path = parser_args.config
        ckpt_path = parser_args.ckpt_path
        topk = parser_args.topk
        device = parser_args.device

        if not parser_args.verbose:
            self.disable_task_loggers()

        task_source = self.get_input_source(parser_args.input)
        task_results = OrderedDict()
        has_exceptions = False

        for id_, input_ in task_source.items():
            try:
                res = self(input_, model_type, cfg_path, ckpt_path, label_file,
                           topk, device)
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
                 model: str='panns_cnn14',
                 config: Optional[os.PathLike]=None,
                 ckpt_path: Optional[os.PathLike]=None,
                 label_file: Optional[os.PathLike]=None,
                 topk: int=1,
                 device: str=paddle.get_device()):
        """
            Python API to call an executor.
        """
        audio_file = os.path.abspath(os.path.expanduser(audio_file))
        paddle.set_device(device)
        self._init_from_path(model, config, ckpt_path, label_file)
        self.preprocess(audio_file)
        self.infer()
        res = self.postprocess(topk)  # Retrieve result of cls.

        return res
