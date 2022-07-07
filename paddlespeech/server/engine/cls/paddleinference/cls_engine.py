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
import os
import time
from collections import OrderedDict
from typing import Optional

import numpy as np
import paddle
import yaml

from paddlespeech.cli.cls.infer import CLSExecutor
from paddlespeech.cli.log import logger
from paddlespeech.resource import CommonTaskResource
from paddlespeech.server.engine.base_engine import BaseEngine
from paddlespeech.server.utils.paddle_predictor import init_predictor
from paddlespeech.server.utils.paddle_predictor import run_model

__all__ = ['CLSEngine', 'PaddleCLSConnectionHandler']


class CLSServerExecutor(CLSExecutor):
    def __init__(self):
        super().__init__()
        self.task_resource = CommonTaskResource(
            task='cls', model_format='static')

    def _init_from_path(
            self,
            model_type: str='panns_cnn14_audioset',
            cfg_path: Optional[os.PathLike]=None,
            model_path: Optional[os.PathLike]=None,
            params_path: Optional[os.PathLike]=None,
            label_file: Optional[os.PathLike]=None,
            predictor_conf: dict=None, ):
        """
        Init model and other resources from a specific path.
        """

        if cfg_path is None or model_path is None or params_path is None or label_file is None:
            tag = model_type + '-' + '32k'
            self.task_resource.set_task_model(model_tag=tag)
            self.res_path = self.task_resource.res_dir
            self.cfg_path = os.path.join(
                self.res_path, self.task_resource.res_dict['cfg_path'])
            self.model_path = os.path.join(
                self.res_path, self.task_resource.res_dict['model_path'])
            self.params_path = os.path.join(
                self.res_path, self.task_resource.res_dict['params_path'])
            self.label_file = os.path.join(
                self.res_path, self.task_resource.res_dict['label_file'])
        else:
            self.cfg_path = os.path.abspath(cfg_path)
            self.model_path = os.path.abspath(model_path)
            self.params_path = os.path.abspath(params_path)
            self.label_file = os.path.abspath(label_file)

        logger.debug(self.cfg_path)
        logger.debug(self.model_path)
        logger.debug(self.params_path)
        logger.debug(self.label_file)

        # config
        with open(self.cfg_path, 'r') as f:
            self._conf = yaml.safe_load(f)
        logger.debug("Read cfg file successfully.")

        # labels
        self._label_list = []
        with open(self.label_file, 'r') as f:
            for line in f:
                self._label_list.append(line.strip())
        logger.debug("Read label file successfully.")

        # Create predictor
        self.predictor_conf = predictor_conf
        self.predictor = init_predictor(
            model_file=self.model_path,
            params_file=self.params_path,
            predictor_conf=self.predictor_conf)
        logger.debug("Create predictor successfully.")

    @paddle.no_grad()
    def infer(self):
        """
        Model inference and result stored in self.output.
        """
        output = run_model(self.predictor, [self._inputs['feats'].numpy()])
        self._outputs['logits'] = output[0]


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
        self.executor = CLSServerExecutor()
        self.config = config
        self.engine_type = "inference"

        try:
            if self.config.predictor_conf.device is not None:
                self.device = self.config.predictor_conf.device
            else:
                self.device = paddle.get_device()
            paddle.set_device(self.device)
        except Exception as e:
            logger.error(
                "Set device failed, please check if device is already used and the parameter 'device' in the yaml file"
            )
            logger.error(e)
            return False

        try:
            self.executor._init_from_path(
                self.config.model_type, self.config.cfg_path,
                self.config.model_path, self.config.params_path,
                self.config.label_file, self.config.predictor_conf)

        except Exception as e:
            logger.error("Initialize CLS server engine Failed.")
            logger.error(e)
            return False

        logger.info("Initialize CLS server engine successfully on device: %s." %
                    (self.device))
        return True


class PaddleCLSConnectionHandler(CLSServerExecutor):
    def __init__(self, cls_engine):
        """The PaddleSpeech CLS Server Connection Handler
           This connection process every cls server request
        Args:
            cls_engine (CLSEngine): The CLS engine
        """
        super().__init__()
        logger.debug(
            "Create PaddleCLSConnectionHandler to process the cls request")

        self._inputs = OrderedDict()
        self._outputs = OrderedDict()
        self.cls_engine = cls_engine
        self.executor = self.cls_engine.executor
        self._conf = self.executor._conf
        self._label_list = self.executor._label_list
        self.predictor = self.executor.predictor

    def run(self, audio_data):
        """engine run 

        Args:
            audio_data (bytes): base64.b64decode
        """

        self.preprocess(io.BytesIO(audio_data))
        st = time.time()
        self.infer()
        infer_time = time.time() - st

        logger.debug("inference time: {}".format(infer_time))
        logger.info("cls engine type: inference")

    def postprocess(self, topk: int):
        """postprocess
        """
        assert topk <= len(
            self._label_list), 'Value of topk is larger than number of labels.'

        result = np.squeeze(self._outputs['logits'], axis=0)
        topk_idx = (-result).argsort()[:topk]
        topk_results = []
        for idx in topk_idx:
            res = {}
            label, score = self._label_list[idx], result[idx]
            res['class_name'] = label
            res['prob'] = score
            topk_results.append(res)

        return topk_results
