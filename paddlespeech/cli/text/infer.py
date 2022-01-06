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
from typing import List
from typing import Optional
from typing import Union

import paddle

from ...s2t.utils.dynamic_import import dynamic_import
from ..executor import BaseExecutor
from ..log import logger
from ..utils import cli_register
from ..utils import download_and_decompress
from ..utils import MODEL_HOME
from ..utils import stats_wrapper

__all__ = ['TextExecutor']

pretrained_models = {
    # The tags for pretrained_models should be "{model_name}[_{dataset}][-{lang}][-...]".
    # e.g. "conformer_wenetspeech-zh-16k", "transformer_aishell-zh-16k" and "panns_cnn6-32k".
    # Command line and python api use "{model_name}[_{dataset}]" as --model, usage:
    # "paddlespeech asr --model conformer_wenetspeech --lang zh --sr 16000 --input ./input.wav"
    "ernie_linear_p7_wudao-punc-zh": {
        'url':
        'https://paddlespeech.bj.bcebos.com/text/ernie_linear_p7_wudao-punc-zh.tar.gz',
        'md5':
        '12283e2ddde1797c5d1e57036b512746',
        'cfg_path':
        'ckpt/model_config.json',
        'ckpt_path':
        'ckpt/model_state.pdparams',
        'vocab_file':
        'punc_vocab.txt',
    },
    "ernie_linear_p3_wudao-punc-zh": {
        'url':
        'https://paddlespeech.bj.bcebos.com/text/ernie_linear_p3_wudao-punc-zh.tar.gz',
        'md5':
        '448eb2fdf85b6a997e7e652e80c51dd2',
        'cfg_path':
        'ckpt/model_config.json',
        'ckpt_path':
        'ckpt/model_state.pdparams',
        'vocab_file':
        'punc_vocab.txt',
    },
}

model_alias = {
    "ernie_linear_p7": "paddlespeech.text.models:ErnieLinear",
    "ernie_linear_p3": "paddlespeech.text.models:ErnieLinear",
}

tokenizer_alias = {
    "ernie_linear_p7": "paddlenlp.transformers:ErnieTokenizer",
    "ernie_linear_p3": "paddlenlp.transformers:ErnieTokenizer",
}


@cli_register(name='paddlespeech.text', description='Text infer command.')
class TextExecutor(BaseExecutor):
    def __init__(self):
        super(TextExecutor, self).__init__()

        self.parser = argparse.ArgumentParser(
            prog='paddlespeech.text', add_help=True)
        self.parser.add_argument(
            '--input', type=str, required=True, help='Input text.')
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
            choices=[tag[:tag.index('-')] for tag in pretrained_models.keys()],
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

    def _get_pretrained_path(self, tag: str) -> os.PathLike:
        """
            Download and returns pretrained resources path of current task.
        """
        assert tag in pretrained_models, 'Can not find pretrained resources of {}.'.format(
            tag)

        res_path = os.path.join(MODEL_HOME, tag)
        decompressed_path = download_and_decompress(pretrained_models[tag],
                                                    res_path)
        decompressed_path = os.path.abspath(decompressed_path)
        logger.info(
            'Use pretrained model stored in: {}'.format(decompressed_path))

        return decompressed_path

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
            logger.info('Model had been initialized.')
            return

        self.task = task

        if cfg_path is None or ckpt_path is None or vocab_file is None:
            tag = '-'.join([model_type, task, lang])
            self.res_path = self._get_pretrained_path(tag)
            self.cfg_path = os.path.join(self.res_path,
                                         pretrained_models[tag]['cfg_path'])
            self.ckpt_path = os.path.join(self.res_path,
                                          pretrained_models[tag]['ckpt_path'])
            self.vocab_file = os.path.join(self.res_path,
                                           pretrained_models[tag]['vocab_file'])
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
            model_class = dynamic_import(model_name, model_alias)
            tokenizer_class = dynamic_import(model_name, tokenizer_alias)
            self.model = model_class(
                cfg_path=self.cfg_path, ckpt_path=self.ckpt_path)
            self.tokenizer = tokenizer_class.from_pretrained('ernie-1.0')
        else:
            raise NotImplementedError

        self.model.eval()

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

    def postprocess(self) -> Union[str, os.PathLike]:
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

        text = parser_args.input
        task = parser_args.task
        model_type = parser_args.model
        lang = parser_args.lang
        cfg_path = parser_args.config
        ckpt_path = parser_args.ckpt_path
        punc_vocab = parser_args.punc_vocab
        device = parser_args.device

        try:
            res = self(text, task, model_type, lang, cfg_path, ckpt_path,
                       punc_vocab, device)
            logger.info('Text Result:\n{}'.format(res))
            return True
        except Exception as e:
            logger.exception(e)
            return False

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
        paddle.set_device(device)
        self._init_from_path(task, model, lang, config, ckpt_path, punc_vocab)
        self.preprocess(text)
        self.infer()
        res = self.postprocess()  # Retrieve result of text task.

        return res
