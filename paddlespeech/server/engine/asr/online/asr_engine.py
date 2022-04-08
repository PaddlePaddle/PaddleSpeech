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
import os
from typing import Optional

import numpy as np
import paddle
from numpy import float32
from yacs.config import CfgNode

from paddlespeech.cli.asr.infer import ASRExecutor
from paddlespeech.cli.log import logger
from paddlespeech.cli.utils import MODEL_HOME
from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.frontend.speech import SpeechSegment
from paddlespeech.s2t.modules.ctc import CTCDecoder
from paddlespeech.s2t.utils.utility import UpdateConfig
from paddlespeech.server.engine.base_engine import BaseEngine
from paddlespeech.server.utils.audio_process import pcm2float
from paddlespeech.server.utils.paddle_predictor import init_predictor

__all__ = ['ASREngine']

pretrained_models = {
    "deepspeech2online_aishell-zh-16k": {
        'url':
        'https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_online_aishell_ckpt_0.1.1.model.tar.gz',
        'md5':
        'd5e076217cf60486519f72c217d21b9b',
        'cfg_path':
        'model.yaml',
        'ckpt_path':
        'exp/deepspeech2_online/checkpoints/avg_1',
        'model':
        'exp/deepspeech2_online/checkpoints/avg_1.jit.pdmodel',
        'params':
        'exp/deepspeech2_online/checkpoints/avg_1.jit.pdiparams',
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

        # init decoder
        cfg = self.config.decode
        decode_batch_size = 1  # for online
        self.decoder.init_decoder(
            decode_batch_size, self.text_feature.vocab_list,
            cfg.decoding_method, cfg.lang_model_path, cfg.alpha, cfg.beta,
            cfg.beam_size, cfg.cutoff_prob, cfg.cutoff_top_n,
            cfg.num_proc_bsearch)

        # init state box
        self.chunk_state_h_box = np.zeros(
            (self.config.num_rnn_layers, 1, self.config.rnn_layer_size),
            dtype=float32)
        self.chunk_state_c_box = np.zeros(
            (self.config.num_rnn_layers, 1, self.config.rnn_layer_size),
            dtype=float32)

    def reset_decoder_and_chunk(self):
        """reset decoder and chunk state for an new audio
        """
        self.decoder.reset_decoder(batch_size=1)
        # init state box, for new audio request
        self.chunk_state_h_box = np.zeros(
            (self.config.num_rnn_layers, 1, self.config.rnn_layer_size),
            dtype=float32)
        self.chunk_state_c_box = np.zeros(
            (self.config.num_rnn_layers, 1, self.config.rnn_layer_size),
            dtype=float32)

    def decode_one_chunk(self, x_chunk, x_chunk_lens, model_type: str):
        """decode one chunk

        Args:
            x_chunk (numpy.array): shape[B, T, D]
            x_chunk_lens (numpy.array): shape[B]
            model_type (str): online model type

        Returns:
            [type]: [description]
        """
        if "deepspeech2online" in model_type:
            input_names = self.am_predictor.get_input_names()
            audio_handle = self.am_predictor.get_input_handle(input_names[0])
            audio_len_handle = self.am_predictor.get_input_handle(
                input_names[1])
            h_box_handle = self.am_predictor.get_input_handle(input_names[2])
            c_box_handle = self.am_predictor.get_input_handle(input_names[3])

            audio_handle.reshape(x_chunk.shape)
            audio_handle.copy_from_cpu(x_chunk)

            audio_len_handle.reshape(x_chunk_lens.shape)
            audio_len_handle.copy_from_cpu(x_chunk_lens)

            h_box_handle.reshape(self.chunk_state_h_box.shape)
            h_box_handle.copy_from_cpu(self.chunk_state_h_box)

            c_box_handle.reshape(self.chunk_state_c_box.shape)
            c_box_handle.copy_from_cpu(self.chunk_state_c_box)

            output_names = self.am_predictor.get_output_names()
            output_handle = self.am_predictor.get_output_handle(output_names[0])
            output_lens_handle = self.am_predictor.get_output_handle(
                output_names[1])
            output_state_h_handle = self.am_predictor.get_output_handle(
                output_names[2])
            output_state_c_handle = self.am_predictor.get_output_handle(
                output_names[3])

            self.am_predictor.run()

            output_chunk_probs = output_handle.copy_to_cpu()
            output_chunk_lens = output_lens_handle.copy_to_cpu()
            self.chunk_state_h_box = output_state_h_handle.copy_to_cpu()
            self.chunk_state_c_box = output_state_c_handle.copy_to_cpu()

            self.decoder.next(output_chunk_probs, output_chunk_lens)
            trans_best, trans_beam = self.decoder.decode()

            return trans_best[0]

        elif "conformer" in model_type or "transformer" in model_type:
            raise Exception("invalid model name")
        else:
            raise Exception("invalid model name")

    def extract_feat(self, samples, sample_rate):
        """extract feat

        Args:
            samples (numpy.array): numpy.float32
            sample_rate (int): sample rate

        Returns:
            x_chunk (numpy.array): shape[B, T, D]
            x_chunk_lens (numpy.array): shape[B]
        """
        # pcm16 -> pcm 32
        samples = pcm2float(samples)

        # read audio
        speech_segment = SpeechSegment.from_pcm(
            samples, sample_rate, transcript=" ")
        # audio augment
        self.collate_fn_test.augmentation.transform_audio(speech_segment)

        # extract speech feature
        spectrum, transcript_part = self.collate_fn_test._speech_featurizer.featurize(
            speech_segment, self.collate_fn_test.keep_transcription_text)
        # CMVN spectrum
        if self.collate_fn_test._normalizer:
            spectrum = self.collate_fn_test._normalizer.apply(spectrum)

        # spectrum augment
        audio = self.collate_fn_test.augmentation.transform_feature(spectrum)

        audio_len = audio.shape[0]
        audio = paddle.to_tensor(audio, dtype='float32')
        # audio_len = paddle.to_tensor(audio_len)
        audio = paddle.unsqueeze(audio, axis=0)

        x_chunk = audio.numpy()
        x_chunk_lens = np.array([audio_len])

        return x_chunk, x_chunk_lens


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
        self.output = ""
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

    def preprocess(self, samples, sample_rate):
        """preprocess

        Args:
            samples (numpy.array): numpy.float32
            sample_rate (int): sample rate

        Returns:
            x_chunk (numpy.array): shape[B, T, D]
            x_chunk_lens (numpy.array): shape[B]
        """
        x_chunk, x_chunk_lens = self.executor.extract_feat(samples, sample_rate)
        return x_chunk, x_chunk_lens

    def run(self, x_chunk, x_chunk_lens, decoder_chunk_size=1):
        """run online engine

        Args:
            x_chunk (numpy.array): shape[B, T, D]
            x_chunk_lens (numpy.array): shape[B]
            decoder_chunk_size(int)
        """
        self.output = self.executor.decode_one_chunk(x_chunk, x_chunk_lens,
                                                     self.config.model_type)

    def postprocess(self):
        """postprocess
        """
        return self.output

    def reset(self):
        """reset engine decoder and inference state
        """
        self.executor.reset_decoder_and_chunk()
        self.output = ""
