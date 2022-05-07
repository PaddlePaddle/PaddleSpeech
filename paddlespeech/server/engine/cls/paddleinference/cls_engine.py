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
from typing import Optional

import numpy as np
import paddle
import yaml

from paddlespeech.cli.cls.infer import CLSExecutor
from paddlespeech.cli.log import logger
from paddlespeech.cli.utils import download_and_decompress
from paddlespeech.cli.utils import MODEL_HOME
from paddlespeech.server.engine.base_engine import BaseEngine
from paddlespeech.server.utils.paddle_predictor import init_predictor
from paddlespeech.server.utils.paddle_predictor import run_model

__all__ = ['CLSEngine']

pretrained_models = {
    "panns_cnn6-32k": {
        'url':
        'https://paddlespeech.bj.bcebos.com/cls/inference_model/panns_cnn6_static.tar.gz',
        'md5':
        'da087c31046d23281d8ec5188c1967da',
        'cfg_path':
        'panns.yaml',
        'model_path':
        'inference.pdmodel',
        'params_path':
        'inference.pdiparams',
        'label_file':
        'audioset_labels.txt',
    },
    "panns_cnn10-32k": {
        'url':
        'https://paddlespeech.bj.bcebos.com/cls/inference_model/panns_cnn10_static.tar.gz',
        'md5':
        '5460cc6eafbfaf0f261cc75b90284ae1',
        'cfg_path':
        'panns.yaml',
        'model_path':
        'inference.pdmodel',
        'params_path':
        'inference.pdiparams',
        'label_file':
        'audioset_labels.txt',
    },
    "panns_cnn14-32k": {
        'url':
        'https://paddlespeech.bj.bcebos.com/cls/inference_model/panns_cnn14_static.tar.gz',
        'md5':
        'ccc80b194821274da79466862b2ab00f',
        'cfg_path':
        'panns.yaml',
        'model_path':
        'inference.pdmodel',
        'params_path':
        'inference.pdiparams',
        'label_file':
        'audioset_labels.txt',
    },
}


class CLSServerExecutor(CLSExecutor):
    def __init__(self):
        super().__init__()
        pass

    def _get_pretrained_path(self, tag: str) -> os.PathLike:
        """
            Download and returns pretrained resources path of current task.
        """
        support_models = list(pretrained_models.keys())
        assert tag in pretrained_models, 'The model "{}" you want to use has not been supported, please choose other models.\nThe support models includes:\n\t\t{}\n'.format(
            tag, '\n\t\t'.join(support_models))

        res_path = os.path.join(MODEL_HOME, tag)
        decompressed_path = download_and_decompress(pretrained_models[tag],
                                                    res_path)
        decompressed_path = os.path.abspath(decompressed_path)
        logger.info(
            'Use pretrained model stored in: {}'.format(decompressed_path))

        return decompressed_path

    def _init_from_path(
            self,
            model_type: str='panns_cnn14',
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
            self.res_path = self._get_pretrained_path(tag)
            self.cfg_path = os.path.join(self.res_path,
                                         pretrained_models[tag]['cfg_path'])
            self.model_path = os.path.join(self.res_path,
                                           pretrained_models[tag]['model_path'])
            self.params_path = os.path.join(
                self.res_path, pretrained_models[tag]['params_path'])
            self.label_file = os.path.join(self.res_path,
                                           pretrained_models[tag]['label_file'])
        else:
            self.cfg_path = os.path.abspath(cfg_path)
            self.model_path = os.path.abspath(model_path)
            self.params_path = os.path.abspath(params_path)
            self.label_file = os.path.abspath(label_file)

        logger.info(self.cfg_path)
        logger.info(self.model_path)
        logger.info(self.params_path)
        logger.info(self.label_file)

        # config
        with open(self.cfg_path, 'r') as f:
            self._conf = yaml.safe_load(f)
        logger.info("Read cfg file successfully.")

        # labels
        self._label_list = []
        with open(self.label_file, 'r') as f:
            for line in f:
                self._label_list.append(line.strip())
        logger.info("Read label file successfully.")

        # Create predictor
        self.predictor_conf = predictor_conf
        self.predictor = init_predictor(
            model_file=self.model_path,
            params_file=self.params_path,
            predictor_conf=self.predictor_conf)
        logger.info("Create predictor successfully.")

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
        self.executor._init_from_path(
            self.config.model_type, self.config.cfg_path,
            self.config.model_path, self.config.params_path,
            self.config.label_file, self.config.predictor_conf)

        logger.info("Initialize CLS server engine successfully.")
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
        logger.info("cls engine type: inference")

    def postprocess(self, topk: int):
        """postprocess
        """
        assert topk <= len(self.executor._label_list
                           ), 'Value of topk is larger than number of labels.'

        result = np.squeeze(self.executor._outputs['logits'], axis=0)
        topk_idx = (-result).argsort()[:topk]
        topk_results = []
        for idx in topk_idx:
            res = {}
            label, score = self.executor._label_list[idx], result[idx]
            res['class_name'] = label
            res['prob'] = score
            topk_results.append(res)

        return topk_results
