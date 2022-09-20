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
import re
from collections import OrderedDict
from typing import List
from typing import Optional
from typing import Union

import paddle
import yaml
from yacs.config import CfgNode

from ..executor import BaseExecutor
from ..log import logger
from ..utils import stats_wrapper
from paddlespeech.text.models.ernie_linear import ErnieLinear

__all__ = ['TextExecutor']


class TextExecutor(BaseExecutor):
    def __init__(self):
        super().__init__(task='text')
        self.parser = argparse.ArgumentParser(
            prog='paddlespeech.text', add_help=True)
        self.parser.add_argument(
            '--input', type=str, default=None, help='Input text.')
        self.parser.add_argument(
            '--task',
            type=str,
            default='punc',
            choices=['punc'],
            help='Choose text task.')
        self.parser.add_argument(
            '--model',
            type=str,
            default='ernie_linear_p7_wudao',
            choices=[
                tag[:tag.index('-')]
                for tag in self.task_resource.pretrained_models.keys()
            ],
            help='Choose model type of text task.')
        self.parser.add_argument(
            '--lang',
            type=str,
            default='zh',
            choices=['zh', 'en'],
            help='Choose model language.')
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
            '--punc_vocab',
            type=str,
            default=None,
            help='Vocabulary file of punctuation restoration task.')
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
                        task: str='punc',
                        model_type: str='ernie_linear_p7_wudao',
                        lang: str='zh',
                        cfg_path: Optional[os.PathLike]=None,
                        ckpt_path: Optional[os.PathLike]=None,
                        vocab_file: Optional[os.PathLike]=None):
        """
            Init model and other resources from a specific path.
        """
        if hasattr(self, 'model'):
            logger.debug('Model had been initialized.')
            return

        self.task = task

        if cfg_path is None or ckpt_path is None or vocab_file is None:
            tag = '-'.join([model_type, task, lang])
            self.task_resource.set_task_model(tag, version=None)
            self.cfg_path = os.path.join(
                self.task_resource.res_dir,
                self.task_resource.res_dict['cfg_path'])
            self.ckpt_path = os.path.join(
                self.task_resource.res_dir,
                self.task_resource.res_dict['ckpt_path'])
            self.vocab_file = os.path.join(
                self.task_resource.res_dir,
                self.task_resource.res_dict['vocab_file'])
        else:
            self.cfg_path = os.path.abspath(cfg_path)
            self.ckpt_path = os.path.abspath(ckpt_path)
            self.vocab_file = os.path.abspath(vocab_file)

        model_name = model_type[:model_type.rindex('_')]
        if self.task == 'punc':
            # punc list
            self._punc_list = []
            with open(self.vocab_file, 'r') as f:
                for line in f:
                    self._punc_list.append(line.strip())

            # model
            model_class, tokenizer_class = self.task_resource.get_model_class(
                model_name)
            self.model = model_class(
                cfg_path=self.cfg_path, ckpt_path=self.ckpt_path)
            self.tokenizer = tokenizer_class.from_pretrained('ernie-1.0')
        else:
            raise NotImplementedError

        self.model.eval()

    #init new models
    def _init_from_path_new(self,
                            task: str='punc',
                            model_type: str='ernie_linear_p7_wudao',
                            lang: str='zh',
                            cfg_path: Optional[os.PathLike]=None,
                            ckpt_path: Optional[os.PathLike]=None,
                            vocab_file: Optional[os.PathLike]=None):
        if hasattr(self, 'model'):
            logger.debug('Model had been initialized.')
            return

        self.task = task

        if cfg_path is None or ckpt_path is None or vocab_file is None:
            tag = '-'.join([model_type, task, lang])
            self.task_resource.set_task_model(tag, version=None)
            self.cfg_path = os.path.join(
                self.task_resource.res_dir,
                self.task_resource.res_dict['cfg_path'])
            self.ckpt_path = os.path.join(
                self.task_resource.res_dir,
                self.task_resource.res_dict['ckpt_path'])
            self.vocab_file = os.path.join(
                self.task_resource.res_dir,
                self.task_resource.res_dict['vocab_file'])
        else:
            self.cfg_path = os.path.abspath(cfg_path)
            self.ckpt_path = os.path.abspath(ckpt_path)
            self.vocab_file = os.path.abspath(vocab_file)

        model_name = model_type[:model_type.rindex('_')]

        if self.task == 'punc':
            # punc list
            self._punc_list = []
            with open(self.vocab_file, 'r') as f:
                for line in f:
                    self._punc_list.append(line.strip())

            # model
            with open(self.cfg_path) as f:
                config = CfgNode(yaml.safe_load(f))
            self.model = ErnieLinear(**config["model"])

            _, tokenizer_class = self.task_resource.get_model_class(model_name)
            state_dict = paddle.load(self.ckpt_path)
            self.model.set_state_dict(state_dict["main_params"])
            self.model.eval()

            #tokenizer: fast version: ernie-3.0-mini-zh   slow version:ernie-1.0
            if 'fast' not in model_type:
                self.tokenizer = tokenizer_class.from_pretrained('ernie-1.0')
            else:
                self.tokenizer = tokenizer_class.from_pretrained(
                    'ernie-3.0-mini-zh')

        else:
            raise NotImplementedError

    def _clean_text(self, text):
        text = text.lower()
        text = re.sub('[^A-Za-z0-9\u4e00-\u9fa5]', '', text)
        text = re.sub(f'[{"".join([p for p in self._punc_list][1:])}]', '',
                      text)
        return text

    def preprocess(self, text: Union[str, os.PathLike]):
        """
            Input preprocess and return paddle.Tensor stored in self.input.
            Input content can be a text(tts), a file(asr, cls) or a streaming(not supported yet).
        """
        if self.task == 'punc':
            clean_text = self._clean_text(text)
            assert len(clean_text) > 0, f'Invalid input string: {text}'

            tokenized_input = self.tokenizer(
                list(clean_text), return_length=True, is_split_into_words=True)

            self._inputs['input_ids'] = tokenized_input['input_ids']
            self._inputs['seg_ids'] = tokenized_input['token_type_ids']
            self._inputs['seq_len'] = tokenized_input['seq_len']
        else:
            raise NotImplementedError

    @paddle.no_grad()
    def infer(self):
        """
            Model inference and result stored in self.output.
        """
        if self.task == 'punc':
            input_ids = paddle.to_tensor(self._inputs['input_ids']).unsqueeze(0)
            seg_ids = paddle.to_tensor(self._inputs['seg_ids']).unsqueeze(0)
            logits, _ = self.model(input_ids, seg_ids)
            preds = paddle.argmax(logits, axis=-1).squeeze(0)

            self._outputs['preds'] = preds
        else:
            raise NotImplementedError

    def postprocess(self, isNewTrainer: bool=False) -> Union[str, os.PathLike]:
        """
            Output postprocess and return human-readable results such as texts and audio files.
        """
        if self.task == 'punc':
            input_ids = self._inputs['input_ids']
            seq_len = self._inputs['seq_len']
            preds = self._outputs['preds']

            tokens = self.tokenizer.convert_ids_to_tokens(
                input_ids[1:seq_len - 1])
            labels = preds[1:seq_len - 1].tolist()
            assert len(tokens) == len(labels)
            if isNewTrainer:
                self._punc_list = [0] + self._punc_list
            text = ''
            for t, l in zip(tokens, labels):
                text += t
                if l != 0:  # Non punc.
                    text += self._punc_list[l]
            return text
        else:
            raise NotImplementedError

    def execute(self, argv: List[str]) -> bool:
        """
            Command line entry.
        """
        parser_args = self.parser.parse_args(argv)

        task = parser_args.task
        model_type = parser_args.model
        lang = parser_args.lang
        cfg_path = parser_args.config
        ckpt_path = parser_args.ckpt_path
        punc_vocab = parser_args.punc_vocab
        device = parser_args.device

        if not parser_args.verbose:
            self.disable_task_loggers()

        task_source = self.get_input_source(parser_args.input)
        task_results = OrderedDict()
        has_exceptions = False

        for id_, input_ in task_source.items():
            try:
                res = self(input_, task, model_type, lang, cfg_path, ckpt_path,
                           punc_vocab, device)
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
    def __call__(
            self,
            text: str,
            task: str='punc',
            model: str='ernie_linear_p7_wudao',
            lang: str='zh',
            config: Optional[os.PathLike]=None,
            ckpt_path: Optional[os.PathLike]=None,
            punc_vocab: Optional[os.PathLike]=None,
            device: str=paddle.get_device(), ):
        """
            Python API to call an executor.
        """
        #Here is old version models 
        if model in ['ernie_linear_p7_wudao', 'ernie_linear_p3_wudao']:
            paddle.set_device(device)
            self._init_from_path(task, model, lang, config, ckpt_path,
                                 punc_vocab)
            self.preprocess(text)
            self.infer()
            res = self.postprocess()  # Retrieve result of text task.
        #Add new way to infer
        else:
            paddle.set_device(device)
            self._init_from_path_new(task, model, lang, config, ckpt_path,
                                     punc_vocab)
            self.preprocess(text)
            self.infer()
            res = self.postprocess(isNewTrainer=True)
        return res

