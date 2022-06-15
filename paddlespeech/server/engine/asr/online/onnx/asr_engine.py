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
import sys
from typing import ByteString
from typing import Optional

import numpy as np
import paddle
from numpy import float32
from yacs.config import CfgNode

from paddlespeech.cli.asr.infer import ASRExecutor
from paddlespeech.cli.log import logger
from paddlespeech.cli.utils import MODEL_HOME
from paddlespeech.resource import CommonTaskResource
from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.modules.ctc import CTCDecoder
from paddlespeech.s2t.transform.transformation import Transformation
from paddlespeech.s2t.utils.utility import UpdateConfig
from paddlespeech.server.engine.base_engine import BaseEngine
from paddlespeech.server.utils import onnx_infer

__all__ = ['PaddleASRConnectionHanddler', 'ASRServerExecutor', 'ASREngine']


# ASR server connection process class
class PaddleASRConnectionHanddler:
    def __init__(self, asr_engine):
        """Init a Paddle ASR Connection Handler instance

        Args:
            asr_engine (ASREngine): the global asr engine
        """
        super().__init__()
        logger.info(
            "create an paddle asr connection handler to process the websocket connection"
        )
        self.config = asr_engine.config  # server config
        self.model_config = asr_engine.executor.config
        self.asr_engine = asr_engine

        # model_type, sample_rate and text_feature is shared for deepspeech2 and conformer
        self.model_type = self.asr_engine.executor.model_type
        self.sample_rate = self.asr_engine.executor.sample_rate
        # tokens to text
        self.text_feature = self.asr_engine.executor.text_feature

        # extract feat, new only fbank in conformer model
        self.preprocess_conf = self.model_config.preprocess_config
        self.preprocess_args = {"train": False}
        self.preprocessing = Transformation(self.preprocess_conf)

        # frame window and frame shift, in samples unit
        self.win_length = self.preprocess_conf.process[0]['win_length']
        self.n_shift = self.preprocess_conf.process[0]['n_shift']

        assert self.preprocess_conf.process[0]['fs'] == self.sample_rate, (
            self.sample_rate, self.preprocess_conf.process[0]['fs'])
        self.frame_shift_in_ms = int(
            self.n_shift / self.preprocess_conf.process[0]['fs'] * 1000)

        self.continuous_decoding = self.config.get("continuous_decoding", False)
        self.init_decoder()
        self.reset()

    def init_decoder(self):
        if "deepspeech2" in self.model_type:
            assert self.continuous_decoding is False, "ds2 model not support endpoint"
            self.am_predictor = self.asr_engine.executor.am_predictor

            self.decoder = CTCDecoder(
                odim=self.model_config.output_dim,  # <blank> is in  vocab
                enc_n_units=self.model_config.rnn_layer_size * 2,
                blank_id=self.model_config.blank_id,
                dropout_rate=0.0,
                reduction=True,  # sum
                batch_average=True,  # sum / batch_size
                grad_norm_type=self.model_config.get('ctc_grad_norm_type',
                                                     None))

            cfg = self.model_config.decode
            decode_batch_size = 1  # for online
            self.decoder.init_decoder(
                decode_batch_size, self.text_feature.vocab_list,
                cfg.decoding_method, cfg.lang_model_path, cfg.alpha, cfg.beta,
                cfg.beam_size, cfg.cutoff_prob, cfg.cutoff_top_n,
                cfg.num_proc_bsearch)
        else:
            raise ValueError(f"Not supported: {self.model_type}")

    def model_reset(self):
        # cache for audio and feat
        self.remained_wav = None
        self.cached_feat = None

    def output_reset(self):
        ## outputs
        # partial/ending decoding results
        self.result_transcripts = ['']

    def reset_continuous_decoding(self):
        """
        when in continous decoding, reset for next utterance.
        """
        self.global_frame_offset = self.num_frames
        self.model_reset()

    def reset(self):
        if "deepspeech2" in self.model_type:
            # for deepspeech2
            # init state
            self.chunk_state_h_box = np.zeros(
                (self.model_config.num_rnn_layers, 1,
                 self.model_config.rnn_layer_size),
                dtype=float32)
            self.chunk_state_c_box = np.zeros(
                (self.model_config.num_rnn_layers, 1,
                 self.model_config.rnn_layer_size),
                dtype=float32)
            self.decoder.reset_decoder(batch_size=1)
        else:
            raise NotImplementedError(f"{self.model_type} not support.")

        self.device = None

        ## common
        # global sample and frame step
        self.num_samples = 0
        self.global_frame_offset = 0
        # frame step of cur utterance
        self.num_frames = 0

        ## endpoint
        self.endpoint_state = False  # True for detect endpoint

        ## conformer
        self.model_reset()

        ## outputs
        self.output_reset()

    def extract_feat(self, samples: ByteString):
        logger.info("Online ASR extract the feat")
        samples = np.frombuffer(samples, dtype=np.int16)
        assert samples.ndim == 1

        self.num_samples += samples.shape[0]
        logger.info(
            f"This package receive {samples.shape[0]} pcm data. Global samples:{self.num_samples}"
        )

        # self.reamined_wav stores all the samples,
        # include the original remained_wav and this package samples
        if self.remained_wav is None:
            self.remained_wav = samples
        else:
            assert self.remained_wav.ndim == 1  # (T,)
            self.remained_wav = np.concatenate([self.remained_wav, samples])
        logger.info(
            f"The concatenation of remain and now audio samples length is: {self.remained_wav.shape}"
        )

        if len(self.remained_wav) < self.win_length:
            # samples not enough for feature window
            return 0

        # fbank
        x_chunk = self.preprocessing(self.remained_wav, **self.preprocess_args)
        x_chunk = paddle.to_tensor(x_chunk, dtype="float32").unsqueeze(axis=0)

        # feature cache
        if self.cached_feat is None:
            self.cached_feat = x_chunk
        else:
            assert (len(x_chunk.shape) == 3)  # (B,T,D)
            assert (len(self.cached_feat.shape) == 3)  # (B,T,D)
            self.cached_feat = paddle.concat(
                [self.cached_feat, x_chunk], axis=1)

        # set the feat device
        if self.device is None:
            self.device = self.cached_feat.place

        # cur frame step
        num_frames = x_chunk.shape[1]

        # global frame step
        self.num_frames += num_frames

        # update remained wav
        self.remained_wav = self.remained_wav[self.n_shift * num_frames:]

        logger.info(
            f"process the audio feature success, the cached feat shape: {self.cached_feat.shape}"
        )
        logger.info(
            f"After extract feat, the cached remain the audio samples: {self.remained_wav.shape}"
        )
        logger.info(f"global samples: {self.num_samples}")
        logger.info(f"global frames: {self.num_frames}")

    def decode(self, is_finished=False):
        """advance decoding

        Args:
            is_finished (bool, optional): Is last frame or not. Defaults to False.

        Returns:
            None: 
        """
        if "deepspeech2" in self.model_type:
            decoding_chunk_size = 1  # decoding chunk size = 1. int decoding frame unit

            context = 7  # context=7, in audio frame unit
            subsampling = 4  # subsampling=4, in audio frame unit

            cached_feature_num = context - subsampling
            # decoding window for model, in audio frame unit
            decoding_window = (decoding_chunk_size - 1) * subsampling + context
            # decoding stride for model, in audio frame unit
            stride = subsampling * decoding_chunk_size

            if self.cached_feat is None:
                logger.info("no audio feat, please input more pcm data")
                return

            num_frames = self.cached_feat.shape[1]
            logger.info(
                f"Required decoding window {decoding_window} frames, and the connection has {num_frames} frames"
            )

            # the cached feat must be larger decoding_window
            if num_frames < decoding_window and not is_finished:
                logger.info(
                    f"frame feat num is less than {decoding_window}, please input more pcm data"
                )
                return None, None

            # if is_finished=True, we need at least context frames
            if num_frames < context:
                logger.info(
                    "flast {num_frames} is less than context {context} frames, and we cannot do model forward"
                )
                return None, None

            logger.info("start to do model forward")
            # num_frames - context + 1 ensure that current frame can get context window
            if is_finished:
                # if get the finished chunk, we need process the last context
                left_frames = context
            else:
                # we only process decoding_window frames for one chunk
                left_frames = decoding_window

            end = None
            for cur in range(0, num_frames - left_frames + 1, stride):
                end = min(cur + decoding_window, num_frames)

                # extract the audio
                x_chunk = self.cached_feat[:, cur:end, :].numpy()
                x_chunk_lens = np.array([x_chunk.shape[1]])

                trans_best = self.decode_one_chunk(x_chunk, x_chunk_lens)

            self.result_transcripts = [trans_best]

            # update feat cache
            self.cached_feat = self.cached_feat[:, end - cached_feature_num:, :]

            # return trans_best[0]
        else:
            raise Exception(f"{self.model_type} not support paddleinference.")

    @paddle.no_grad()
    def decode_one_chunk(self, x_chunk, x_chunk_lens):
        """forward one chunk frames

        Args:
            x_chunk (np.ndarray): (B,T,D), audio frames.
            x_chunk_lens ([type]): (B,), audio frame lens

        Returns:
            logprob: poster probability.
        """
        logger.info("start to decoce one chunk for deepspeech2")
        # state_c, state_h, audio_lens, audio
        # 'chunk_state_c_box', 'chunk_state_h_box', 'audio_chunk_lens', 'audio_chunk'
        input_names = [n.name for n in self.am_predictor.get_inputs()]
        logger.info(f"ort inputs: {input_names}")
        # 'softmax_0.tmp_0', 'tmp_5', 'concat_0.tmp_0', 'concat_1.tmp_0'
        # audio, audio_lens, state_h, state_c
        output_names = [n.name for n in self.am_predictor.get_outputs()]
        logger.info(f"ort outpus: {output_names}")
        assert (len(input_names) == len(output_names))
        assert isinstance(input_names[0], str)

        input_datas = [self.chunk_state_c_box, self.chunk_state_h_box, x_chunk_lens, x_chunk]
        feeds = dict(zip(input_names, input_datas))

        outputs = self.am_predictor.run(
            [*output_names],
            {**feeds})

        output_chunk_probs, output_chunk_lens, self.chunk_state_h_box, self.chunk_state_c_box = outputs
        self.decoder.next(output_chunk_probs, output_chunk_lens)
        trans_best, trans_beam = self.decoder.decode()
        logger.info(f"decode one best result for deepspeech2: {trans_best[0]}")
        return trans_best[0]

    def get_result(self):
        """return partial/ending asr result.

        Returns:
            str: one best result of partial/ending.
        """
        if len(self.result_transcripts) > 0:
            return self.result_transcripts[0]
        else:
            return ''


class ASRServerExecutor(ASRExecutor):
    def __init__(self):
        super().__init__()
        self.task_resource = CommonTaskResource(
            task='asr', model_format='static', inference_mode='online')

    def update_config(self) -> None:
        if "deepspeech2" in self.model_type:
            with UpdateConfig(self.config):
                # download lm
                self.config.decode.lang_model_path = os.path.join(
                    MODEL_HOME, 'language_model',
                    self.config.decode.lang_model_path)

            lm_url = self.task_resource.res_dict['lm_url']
            lm_md5 = self.task_resource.res_dict['lm_md5']
            logger.info(f"Start to load language model {lm_url}")
            self.download_lm(
                lm_url,
                os.path.dirname(self.config.decode.lang_model_path), lm_md5)
        else:
            raise NotImplementedError(
                f"{self.model_type} not support paddleinference.")

    def init_model(self) -> None:

        if "deepspeech2" in self.model_type:
            # AM predictor
            logger.info("ASR engine start to init the am predictor")
            self.am_predictor = onnx_infer.get_sess(
                model_path=self.am_model, sess_conf=self.am_predictor_conf)
        else:
            raise NotImplementedError(
                f"{self.model_type} not support paddleinference.")

    def _init_from_path(self,
                        model_type: str=None,
                        am_model: Optional[os.PathLike]=None,
                        am_params: Optional[os.PathLike]=None,
                        lang: str='zh',
                        sample_rate: int=16000,
                        cfg_path: Optional[os.PathLike]=None,
                        decode_method: str='attention_rescoring',
                        num_decoding_left_chunks: int=-1,
                        am_predictor_conf: dict=None):
        """
        Init model and other resources from a specific path.
        """
        if not model_type or not lang or not sample_rate:
            logger.error(
                "The model type or lang or sample rate is None, please input an valid server parameter yaml"
            )
            return False
        assert am_params is None, "am_params not used in onnx engine"

        self.model_type = model_type
        self.sample_rate = sample_rate
        self.decode_method = decode_method
        self.num_decoding_left_chunks = num_decoding_left_chunks
        # conf for paddleinference predictor or onnx
        self.am_predictor_conf = am_predictor_conf
        logger.info(f"model_type: {self.model_type}")

        sample_rate_str = '16k' if sample_rate == 16000 else '8k'
        tag = model_type + '-' + lang + '-' + sample_rate_str
        self.task_resource.set_task_model(model_tag=tag)

        if cfg_path is None:
            self.res_path = self.task_resource.res_dir
            self.cfg_path = os.path.join(
                self.res_path, self.task_resource.res_dict['cfg_path'])
        else:
            self.cfg_path = os.path.abspath(cfg_path)
            self.res_path = os.path.dirname(
                os.path.dirname(os.path.abspath(self.cfg_path)))

        self.am_model = os.path.join(self.res_path,
                                         self.task_resource.res_dict['model']) if am_model is None else os.path.abspath(am_model)
        self.am_params = os.path.join(self.res_path,
                                          self.task_resource.res_dict['params']) if am_params is None else os.path.abspath(am_params)

        logger.info("Load the pretrained model:")
        logger.info(f"  tag = {tag}")
        logger.info(f"  res_path: {self.res_path}")
        logger.info(f"  cfg path: {self.cfg_path}")
        logger.info(f"  am_model path: {self.am_model}")
        logger.info(f"  am_params path: {self.am_params}")

        #Init body.
        self.config = CfgNode(new_allowed=True)
        self.config.merge_from_file(self.cfg_path)

        if self.config.spm_model_prefix:
            self.config.spm_model_prefix = os.path.join(
                self.res_path, self.config.spm_model_prefix)
            logger.info(f"spm model path: {self.config.spm_model_prefix}")

        self.vocab = self.config.vocab_filepath

        self.text_feature = TextFeaturizer(
            unit_type=self.config.unit_type,
            vocab=self.config.vocab_filepath,
            spm_model_prefix=self.config.spm_model_prefix)

        self.update_config()

        # AM predictor
        self.init_model()

        logger.info(f"create the {model_type} model success")
        return True


class ASREngine(BaseEngine):
    """ASR model resource

    Args:
        metaclass: Defaults to Singleton.
    """

    def __init__(self):
        super(ASREngine, self).__init__()

    def init_model(self) -> bool:
        if not self.executor._init_from_path(
                model_type=self.config.model_type,
                am_model=self.config.am_model,
                am_params=self.config.am_params,
                lang=self.config.lang,
                sample_rate=self.config.sample_rate,
                cfg_path=self.config.cfg_path,
                decode_method=self.config.decode_method,
                num_decoding_left_chunks=self.config.num_decoding_left_chunks,
                am_predictor_conf=self.config.am_predictor_conf):
            return False
        return True

    def init(self, config: dict) -> bool:
        """init engine resource

        Args:
            config_file (str): config file

        Returns:
            bool: init failed or success
        """
        self.config = config
        self.executor = ASRServerExecutor()

        try:
            self.device = self.config.get("device", paddle.get_device())
            paddle.set_device(self.device)
        except BaseException as e:
            logger.error(
                f"Set device failed, please check if device '{self.device}' is already used and the parameter 'device' in the yaml file"
            )
            logger.error(
                "If all GPU or XPU is used, you can set the server to 'cpu'")
            sys.exit(-1)

        logger.info(f"paddlespeech_server set the device: {self.device}")

        if not self.init_model():
            logger.error(
                "Init the ASR server occurs error, please check the server configuration yaml"
            )
            return False

        logger.info("Initialize ASR server engine successfully.")
        return True

    def new_handler(self):
        """New handler from model.

        Returns:
            PaddleASRConnectionHanddler: asr handler instance
        """
        return PaddleASRConnectionHanddler(self)

    def preprocess(self, *args, **kwargs):
        raise NotImplementedError("Online not using this.")

    def run(self, *args, **kwargs):
        raise NotImplementedError("Online not using this.")

    def postprocess(self):
        raise NotImplementedError("Online not using this.")
