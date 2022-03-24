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
import io
import time
from typing import List

import paddle

from paddlespeech.cli.cls.infer import CLSExecutor
from paddlespeech.cli.log import logger
from paddlespeech.server.engine.base_engine import BaseEngine

__all__ = ['CLSEngine']


class CLSServerExecutor(CLSExecutor):
    def __init__(self):
        super().__init__()
        pass

    def get_topk_results(self, topk: int) -> List:
        assert topk <= len(
            self._label_list), 'Value of topk is larger than number of labels.'

        result = self._outputs['logits'].squeeze(0).numpy()
        topk_idx = (-result).argsort()[:topk]
        res = {}
        topk_results = []
        for idx in topk_idx:
            label, score = self._label_list[idx], result[idx]
            res['class'] = label
            res['prob'] = score
            topk_results.append(res)
        return topk_results


class CLSEngine(BaseEngine):
    """CLS server engine

    Args:
        metaclass: Defaults to Singleton.
    """

    def __init__(self):
        super(CLSEngine, self).__init__()

    def init(self, config: dict) -> bool:
        """init engine resource

        Args:
            config_file (str): config file

        Returns:
            bool: init failed or success
        """
        self.input = None
        self.output = None
        self.executor = CLSServerExecutor()
        self.config = config
        try:
            if self.config.device:
                self.device = self.config.device
            else:
                self.device = paddle.get_device()
            paddle.set_device(self.device)
        except BaseException:
            logger.error(
                "Set device failed, please check if device is already used and the parameter 'device' in the yaml file"
            )

        try:
            self.executor._init_from_path(
                self.config.model, self.config.cfg_path, self.config.ckpt_path,
                self.config.label_file)
        except BaseException:
            logger.error("Initialize CLS server engine Failed.")
            return False

        logger.info("Initialize CLS server engine successfully on device: %s." %
                    (self.device))
        return True

    def run(self, audio_data):
        """engine run 

        Args:
            audio_data (bytes): base64.b64decode
        """
        self.executor.preprocess(io.BytesIO(audio_data))
        st = time.time()
        self.executor.infer()
        infer_time = time.time() - st

        logger.info("inference time: {}".format(infer_time))
        logger.info("cls engine type: python")

    def postprocess(self, topk: int):
        """postprocess
        """
        assert topk <= len(self.executor._label_list
                           ), 'Value of topk is larger than number of labels.'

        result = self.executor._outputs['logits'].squeeze(0).numpy()
        topk_idx = (-result).argsort()[:topk]
        topk_results = []
        for idx in topk_idx:
            res = {}
            label, score = self.executor._label_list[idx], result[idx]
            res['class_name'] = label
            res['prob'] = score
            topk_results.append(res)

        return topk_results
