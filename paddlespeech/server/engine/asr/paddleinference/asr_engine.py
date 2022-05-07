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

import paddle
from yacs.config import CfgNode

from paddlespeech.cli.asr.infer import ASRExecutor
from paddlespeech.cli.log import logger
from paddlespeech.cli.utils import MODEL_HOME
from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.modules.ctc import CTCDecoder
from paddlespeech.s2t.utils.utility import UpdateConfig
from paddlespeech.server.engine.base_engine import BaseEngine
from paddlespeech.server.utils.paddle_predictor import init_predictor
from paddlespeech.server.utils.paddle_predictor import run_model

__all__ = ['ASREngine']

pretrained_models = {
    "deepspeech2offline_aishell-zh-16k": {
        'url':
        'https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_aishell_ckpt_0.1.1.model.tar.gz',
        'md5':
        '932c3593d62fe5c741b59b31318aa314',
        'cfg_path':
        'model.yaml',
        'ckpt_path':
        'exp/deepspeech2/checkpoints/avg_1',
        'model':
        'exp/deepspeech2/checkpoints/avg_1.jit.pdmodel',
        'params':
        'exp/deepspeech2/checkpoints/avg_1.jit.pdiparams',
        'lm_url':
        'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm',
        'lm_md5':
        '29e02312deb2e59b3c8686c7966d4fe3'
    },
}


class ASRServerExecutor(ASRExecutor):
    def __init__(self):
        super().__init__()
        pass

    def _init_from_path(self,
                        model_type: str='wenetspeech',
                        am_model: Optional[os.PathLike]=None,
                        am_params: Optional[os.PathLike]=None,
                        lang: str='zh',
                        sample_rate: int=16000,
                        cfg_path: Optional[os.PathLike]=None,
                        decode_method: str='attention_rescoring',
                        am_predictor_conf: dict=None):
        """
        Init model and other resources from a specific path.
        """

        if cfg_path is None or am_model is None or am_params is None:
            sample_rate_str = '16k' if sample_rate == 16000 else '8k'
            tag = model_type + '-' + lang + '-' + sample_rate_str
            res_path = self._get_pretrained_path(tag)  # wenetspeech_zh
            self.res_path = res_path
            self.cfg_path = os.path.join(res_path,
                                         pretrained_models[tag]['cfg_path'])

            self.am_model = os.path.join(res_path,
                                         pretrained_models[tag]['model'])
            self.am_params = os.path.join(res_path,
                                          pretrained_models[tag]['params'])
            logger.info(res_path)
            logger.info(self.cfg_path)
            logger.info(self.am_model)
            logger.info(self.am_params)
        else:
            self.cfg_path = os.path.abspath(cfg_path)
            self.am_model = os.path.abspath(am_model)
            self.am_params = os.path.abspath(am_params)
            self.res_path = os.path.dirname(
                os.path.dirname(os.path.abspath(self.cfg_path)))

        #Init body.
        self.config = CfgNode(new_allowed=True)
        self.config.merge_from_file(self.cfg_path)

        with UpdateConfig(self.config):
            if "deepspeech2online" in model_type or "deepspeech2offline" in model_type:
                from paddlespeech.s2t.io.collator import SpeechCollator
                self.vocab = self.config.vocab_filepath
                self.config.decode.lang_model_path = os.path.join(
                    MODEL_HOME, 'language_model',
                    self.config.decode.lang_model_path)
                self.collate_fn_test = SpeechCollator.from_config(self.config)
                self.text_feature = TextFeaturizer(
                    unit_type=self.config.unit_type, vocab=self.vocab)

                lm_url = pretrained_models[tag]['lm_url']
                lm_md5 = pretrained_models[tag]['lm_md5']
                self.download_lm(
                    lm_url,
                    os.path.dirname(self.config.decode.lang_model_path), lm_md5)
            elif "conformer" in model_type or "transformer" in model_type or "wenetspeech" in model_type:
                raise Exception("wrong type")
            else:
                raise Exception("wrong type")

        # AM predictor
        self.am_predictor_conf = am_predictor_conf
        self.am_predictor = init_predictor(
            model_file=self.am_model,
            params_file=self.am_params,
            predictor_conf=self.am_predictor_conf)

        # decoder
        self.decoder = CTCDecoder(
            odim=self.config.output_dim,  # <blank> is in  vocab
            enc_n_units=self.config.rnn_layer_size * 2,
            blank_id=self.config.blank_id,
            dropout_rate=0.0,
            reduction=True,  # sum
            batch_average=True,  # sum / batch_size
            grad_norm_type=self.config.get('ctc_grad_norm_type', None))

    @paddle.no_grad()
    def infer(self, model_type: str):
        """
        Model inference and result stored in self.output.
        """
        cfg = self.config.decode
        audio = self._inputs["audio"]
        audio_len = self._inputs["audio_len"]
        if "deepspeech2online" in model_type or "deepspeech2offline" in model_type:
            decode_batch_size = audio.shape[0]
            # init once
            self.decoder.init_decoder(
                decode_batch_size, self.text_feature.vocab_list,
                cfg.decoding_method, cfg.lang_model_path, cfg.alpha, cfg.beta,
                cfg.beam_size, cfg.cutoff_prob, cfg.cutoff_top_n,
                cfg.num_proc_bsearch)

            output_data = run_model(self.am_predictor,
                                    [audio.numpy(), audio_len.numpy()])

            probs = output_data[0]
            eouts_len = output_data[1]

            batch_size = probs.shape[0]
            self.decoder.reset_decoder(batch_size=batch_size)
            self.decoder.next(probs, eouts_len)
            trans_best, trans_beam = self.decoder.decode()

            # self.model.decoder.del_decoder()
            self._outputs["result"] = trans_best[0]

        elif "conformer" in model_type or "transformer" in model_type:
            raise Exception("invalid model name")
        else:
            raise Exception("invalid model name")


class ASREngine(BaseEngine):
    """ASR server engine

    Args:
        metaclass: Defaults to Singleton.
    """

    def __init__(self):
        super(ASREngine, self).__init__()

    def init(self, config: dict) -> bool:
        """init engine resource

        Args:
            config_file (str): config file

        Returns:
            bool: init failed or success
        """
        self.input = None
        self.output = None
        self.executor = ASRServerExecutor()
        self.config = config

        self.executor._init_from_path(
            model_type=self.config.model_type,
            am_model=self.config.am_model,
            am_params=self.config.am_params,
            lang=self.config.lang,
            sample_rate=self.config.sample_rate,
            cfg_path=self.config.cfg_path,
            decode_method=self.config.decode_method,
            am_predictor_conf=self.config.am_predictor_conf)

        logger.info("Initialize ASR server engine successfully.")
        return True

    def run(self, audio_data):
        """engine run 

        Args:
            audio_data (bytes): base64.b64decode
        """
        if self.executor._check(
                io.BytesIO(audio_data), self.config.sample_rate,
                self.config.force_yes):
            logger.info("start running asr engine")
            self.executor.preprocess(self.config.model_type,
                                     io.BytesIO(audio_data))
            st = time.time()
            self.executor.infer(self.config.model_type)
            infer_time = time.time() - st
            self.output = self.executor.postprocess()  # Retrieve result of asr.
            logger.info("end inferring asr engine")
        else:
            logger.info("file check failed!")
            self.output = None

        logger.info("inference time: {}".format(infer_time))
        logger.info("asr engine type: paddle inference")

    def postprocess(self):
        """postprocess
        """
        return self.output
