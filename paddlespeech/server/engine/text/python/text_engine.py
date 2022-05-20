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
from collections import OrderedDict

import paddle

from paddlespeech.cli.log import logger
from paddlespeech.cli.text.infer import TextExecutor
from paddlespeech.server.engine.base_engine import BaseEngine


class PaddleTextConnectionHandler:
    def __init__(self, text_engine):
        """The PaddleSpeech Text Server Connection Handler
           This connection process every server request
        Args:
            text_engine (TextEngine): The Text engine
        """
        super().__init__()
        logger.info(
            "Create PaddleTextConnectionHandler to process the text request")
        self.text_engine = text_engine
        self.task = self.text_engine.executor.task
        self.model = self.text_engine.executor.model
        self.tokenizer = self.text_engine.executor.tokenizer
        self._punc_list = self.text_engine.executor._punc_list
        self._inputs = OrderedDict()
        self._outputs = OrderedDict()

    @paddle.no_grad()
    def run(self, text):
        """The connection process the request text

        Args:
            text (str): the request text

        Returns:
            str: the punctuation text
        """
        self.preprocess(text)
        self.infer()
        res = self.postprocess()

        return res

    @paddle.no_grad()
    def preprocess(self, text):
        """
            Input preprocess and return paddle.Tensor stored in self.input.
            Input content can be a text(tts), a file(asr, cls) or a streaming(not supported yet).

        Args:
            text (str): the request text
        """
        if self.task == 'punc':
            clean_text = self.text_engine.executor._clean_text(text)
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
        """Model inference and result stored in self.output.
        """
        if self.task == 'punc':
            input_ids = paddle.to_tensor(self._inputs['input_ids']).unsqueeze(0)
            seg_ids = paddle.to_tensor(self._inputs['seg_ids']).unsqueeze(0)
            logits, _ = self.model(input_ids, seg_ids)
            preds = paddle.argmax(logits, axis=-1).squeeze(0)

            self._outputs['preds'] = preds
        else:
            raise NotImplementedError

    def postprocess(self):
        """Output postprocess and return human-readable results such as texts and audio files.

        Returns:
            str: The punctuation text
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


class TextServerExecutor(TextExecutor):
    def __init__(self):
        """The wrapper for TextEcutor
        """
        super().__init__()
        pass


class TextEngine(BaseEngine):
    def __init__(self):
        """The Text Engine
        """
        super(TextEngine, self).__init__()
        logger.info("Create the TextEngine Instance")

    def init(self, config: dict):
        """Init the Text Engine

        Args:
            config (dict): The server configuation

        Returns:
            bool: The engine instance flag
        """
        logger.info("Init the text engine")
        try:
            self.config = config
            if self.config.device:
                self.device = self.config.device
            else:
                self.device = paddle.get_device()

            paddle.set_device(self.device)
            logger.info(f"Text Engine set the device: {self.device}")
        except BaseException as e:
            logger.error(
                "Set device failed, please check if device is already used and the parameter 'device' in the yaml file"
            )
            logger.error("Initialize Text server engine Failed on device: %s." %
                         (self.device))
            return False

        self.executor = TextServerExecutor()
        self.executor._init_from_path(
            task=config.task,
            model_type=config.model_type,
            lang=config.lang,
            cfg_path=config.cfg_path,
            ckpt_path=config.ckpt_path,
            vocab_file=config.vocab_file)

        logger.info("Init the text engine successfully")
        return True
