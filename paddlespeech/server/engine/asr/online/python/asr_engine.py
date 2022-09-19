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

from paddlespeech.audio.transform.transformation import Transformation
from paddlespeech.cli.asr.infer import ASRExecutor
from paddlespeech.cli.log import logger
from paddlespeech.resource import CommonTaskResource
from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.modules.ctc import CTCDecoder
from paddlespeech.s2t.utils.tensor_utils import add_sos_eos
from paddlespeech.s2t.utils.tensor_utils import pad_sequence
from paddlespeech.s2t.utils.utility import UpdateConfig
from paddlespeech.server.engine.asr.online.ctc_endpoint import OnlineCTCEndpoingOpt
from paddlespeech.server.engine.asr.online.ctc_endpoint import OnlineCTCEndpoint
from paddlespeech.server.engine.asr.online.ctc_search import CTCPrefixBeamSearch
from paddlespeech.server.engine.base_engine import BaseEngine
from paddlespeech.server.utils.paddle_predictor import init_predictor
from paddlespeech.utils.env import MODEL_HOME

__all__ = ['PaddleASRConnectionHanddler', 'ASRServerExecutor', 'ASREngine']


# ASR server connection process class
class PaddleASRConnectionHanddler:
    def __init__(self, asr_engine):
        """Init a Paddle ASR Connection Handler instance

        Args:
            asr_engine (ASREngine): the global asr engine
        """
        super().__init__()
        logger.debug(
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

        elif "conformer" in self.model_type or "transformer" in self.model_type:
            # acoustic model
            self.model = self.asr_engine.executor.model
            self.continuous_decoding = self.config.continuous_decoding
            logger.debug(f"continue decoding: {self.continuous_decoding}")

            # ctc decoding config
            self.ctc_decode_config = self.asr_engine.executor.config.decode
            self.searcher = CTCPrefixBeamSearch(self.ctc_decode_config)

            # ctc endpoint
            self.endpoint_opt = OnlineCTCEndpoingOpt(
                frame_shift_in_ms=self.frame_shift_in_ms, blank=0)
            self.endpointer = OnlineCTCEndpoint(self.endpoint_opt)
        else:
            raise ValueError(f"Not supported: {self.model_type}")

    def model_reset(self):
        # cache for audio and feat
        self.remained_wav = None
        self.cached_feat = None

        if "deepspeech2" in self.model_type:
            return

        ## conformer
        # cache for conformer online
        self.att_cache = paddle.zeros([0, 0, 0, 0])
        self.cnn_cache = paddle.zeros([0, 0, 0, 0])

        self.encoder_out = None
        # conformer decoding state
        self.offset = 0  # global offset in decoding frame unit

        ## just for record info
        self.chunk_num = 0  # global decoding chunk num, not used

    def output_reset(self):
        ## outputs
        # partial/ending decoding results
        self.result_transcripts = ['']
        # token timestamp result
        self.word_time_stamp = []

        ## just for record
        self.hyps = []

        # one best timestamp viterbi prob is large.
        self.time_stamp = []

    def reset_continuous_decoding(self):
        """
        when in continous decoding, reset for next utterance.
        """
        self.global_frame_offset = self.num_frames
        self.model_reset()
        self.searcher.reset()
        self.endpointer.reset()

        # reset hys will trancate history transcripts.
        # self.output_reset()

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

        if "conformer" in self.model_type or "transformer" in self.model_type:
            self.searcher.reset()
            self.endpointer.reset()

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
        logger.debug(
            f"This package receive {samples.shape[0]} pcm data. Global samples:{self.num_samples}"
        )

        # self.reamined_wav stores all the samples,
        # include the original remained_wav and this package samples
        if self.remained_wav is None:
            self.remained_wav = samples
        else:
            assert self.remained_wav.ndim == 1  # (T,)
            self.remained_wav = np.concatenate([self.remained_wav, samples])
        logger.debug(
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

        logger.debug(
            f"process the audio feature success, the cached feat shape: {self.cached_feat.shape}"
        )
        logger.debug(
            f"After extract feat, the cached remain the audio samples: {self.remained_wav.shape}"
        )
        logger.debug(f"global samples: {self.num_samples}")
        logger.debug(f"global frames: {self.num_frames}")

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
                logger.debug("no audio feat, please input more pcm data")
                return

            num_frames = self.cached_feat.shape[1]
            logger.debug(
                f"Required decoding window {decoding_window} frames, and the connection has {num_frames} frames"
            )

            # the cached feat must be larger decoding_window
            if num_frames < decoding_window and not is_finished:
                logger.debug(
                    f"frame feat num is less than {decoding_window}, please input more pcm data"
                )
                return None, None

            # if is_finished=True, we need at least context frames
            if num_frames < context:
                logger.debug(
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
        elif "conformer" in self.model_type or "transformer" in self.model_type:
            try:
                logger.info(
                    f"we will use the transformer like model : {self.model_type}"
                )
                self.advance_decoding(is_finished)
                self.update_result()

            except Exception as e:
                logger.exception(e)
        else:
            raise Exception("invalid model name")

    @paddle.no_grad()
    def decode_one_chunk(self, x_chunk, x_chunk_lens):
        """forward one chunk frames

        Args:
            x_chunk (np.ndarray): (B,T,D), audio frames.
            x_chunk_lens ([type]): (B,), audio frame lens

        Returns:
            logprob: poster probability.
        """
        logger.debug("start to decoce one chunk for deepspeech2")
        input_names = self.am_predictor.get_input_names()
        audio_handle = self.am_predictor.get_input_handle(input_names[0])
        audio_len_handle = self.am_predictor.get_input_handle(input_names[1])
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
        logger.debug(f"decode one best result for deepspeech2: {trans_best[0]}")
        return trans_best[0]

    @paddle.no_grad()
    def advance_decoding(self, is_finished=False):
        if "deepspeech" in self.model_type:
            return

        # reset endpiont state
        self.endpoint_state = False

        logger.debug(
            "Conformer/Transformer: start to decode with advanced_decoding method"
        )
        cfg = self.ctc_decode_config

        # cur chunk size, in decoding frame unit, e.g. 16
        decoding_chunk_size = cfg.decoding_chunk_size
        # using num of history chunks, e.g -1
        num_decoding_left_chunks = cfg.num_decoding_left_chunks
        assert decoding_chunk_size > 0

        # e.g. 4
        subsampling = self.model.encoder.embed.subsampling_rate
        # e.g. 7
        context = self.model.encoder.embed.right_context + 1

        # processed chunk feature cached for next chunk, e.g. 3
        cached_feature_num = context - subsampling

        # decoding window, in audio frame unit
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        # decoding stride, in audio frame unit
        stride = subsampling * decoding_chunk_size

        if self.cached_feat is None:
            logger.debug("no audio feat, please input more pcm data")
            return

        # (B=1,T,D)
        num_frames = self.cached_feat.shape[1]
        logger.debug(
            f"Required decoding window {decoding_window} frames, and the connection has {num_frames} frames"
        )

        # the cached feat must be larger decoding_window
        if num_frames < decoding_window and not is_finished:
            logger.debug(
                f"frame feat num is less than {decoding_window}, please input more pcm data"
            )
            return None, None

        # if is_finished=True, we need at least context frames
        if num_frames < context:
            logger.debug(
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

        # hist of chunks, in deocding frame unit
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks

        # record the end for removing the processed feat
        outputs = []
        end = None
        for cur in range(0, num_frames - left_frames + 1, stride):
            end = min(cur + decoding_window, num_frames)

            # global chunk_num
            self.chunk_num += 1
            # cur chunk
            chunk_xs = self.cached_feat[:, cur:end, :]
            # forward chunk
            (y, self.att_cache,
             self.cnn_cache) = self.model.encoder.forward_chunk(
                 chunk_xs,
                 self.offset,
                 required_cache_size,
                 att_cache=self.att_cache,
                 cnn_cache=self.cnn_cache)
            outputs.append(y)

            # update the global offset, in decoding frame unit
            self.offset += y.shape[1]

        ys = paddle.cat(outputs, 1)
        if self.encoder_out is None:
            self.encoder_out = ys
        else:
            self.encoder_out = paddle.concat([self.encoder_out, ys], axis=1)
        logger.debug(
            f"This connection handler encoder out shape: {self.encoder_out.shape}"
        )

        # get the ctc probs
        ctc_probs = self.model.ctc.log_softmax(ys)  # (1, maxlen, vocab_size)
        ctc_probs = ctc_probs.squeeze(0)

        ## decoding
        # advance decoding
        self.searcher.search(ctc_probs, self.cached_feat.place)
        # get one best hyps
        self.hyps = self.searcher.get_one_best_hyps()

        # endpoint
        if not is_finished:

            def contain_nonsilence():
                return len(self.hyps) > 0 and len(self.hyps[0]) > 0

            decoding_something = contain_nonsilence()
            if self.endpointer.endpoint_detected(ctc_probs.numpy(),
                                                 decoding_something):
                self.endpoint_state = True
                logger.debug(
                    f"Endpoint is detected at {self.num_frames} frame.")

        # advance cache of feat
        assert self.cached_feat.shape[0] == 1  #(B=1,T,D)
        assert end >= cached_feature_num
        self.cached_feat = self.cached_feat[:, end - cached_feature_num:, :]
        assert len(
            self.cached_feat.shape
        ) == 3, f"current cache feat shape is: {self.cached_feat.shape}"

    def update_result(self):
        """Conformer/Transformer hyps to result.
        """
        logger.debug("update the final result")
        hyps = self.hyps

        # output results and tokenids
        self.result_transcripts = [
            self.text_feature.defeaturize(hyp) for hyp in hyps
        ]
        self.result_tokenids = [hyp for hyp in hyps]

    def get_result(self):
        """return partial/ending asr result.

        Returns:
            str: one best result of partial/ending.
        """
        if len(self.result_transcripts) > 0:
            return self.result_transcripts[0]
        else:
            return ''

    def get_word_time_stamp(self):
        """return token timestamp result.

        Returns:
            list: List of ('w':token, 'bg':time, 'ed':time)
        """
        return self.word_time_stamp

    @paddle.no_grad()
    def rescoring(self):
        """Second-Pass Decoding,
        only for conformer and transformer model.
        """
        if "deepspeech2" in self.model_type:
            logger.debug("deepspeech2 not support rescoring decoding.")
            return

        if "attention_rescoring" != self.ctc_decode_config.decoding_method:
            logger.debug(
                f"decoding method not match: {self.ctc_decode_config.decoding_method}, need attention_rescoring"
            )
            return

        logger.debug("rescoring the final result")

        # last decoding for last audio
        self.searcher.finalize_search()
        # update beam search results
        self.update_result()

        beam_size = self.ctc_decode_config.beam_size
        hyps = self.searcher.get_hyps()
        if hyps is None or len(hyps) == 0:
            logger.info("No Hyps!")
            return

        # rescore by decoder post probability

        # assert len(hyps) == beam_size
        # list of Tensor
        hyp_list = []
        for hyp in hyps:
            hyp_content = hyp[0]
            # Prevent the hyp is empty
            if len(hyp_content) == 0:
                hyp_content = (self.model.ctc.blank_id, )

            hyp_content = paddle.to_tensor(
                hyp_content, place=self.device, dtype=paddle.long)
            hyp_list.append(hyp_content)

        hyps_pad = pad_sequence(
            hyp_list, batch_first=True, padding_value=self.model.ignore_id)
        hyps_lens = paddle.to_tensor(
            [len(hyp[0]) for hyp in hyps], place=self.device,
            dtype=paddle.long)  # (beam_size,)
        hyps_pad, _ = add_sos_eos(hyps_pad, self.model.sos, self.model.eos,
                                  self.model.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining

        encoder_out = self.encoder_out.repeat(beam_size, 1, 1)
        encoder_mask = paddle.ones(
            (beam_size, 1, encoder_out.shape[1]), dtype=paddle.bool)
        decoder_out, _ = self.model.decoder(
            encoder_out, encoder_mask, hyps_pad,
            hyps_lens)  # (beam_size, max_hyps_len, vocab_size)
        # ctc score in ln domain
        decoder_out = paddle.nn.functional.log_softmax(decoder_out, axis=-1)
        decoder_out = decoder_out.numpy()

        # Only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0
        # hyps is List[(Text=List[int], Score=float)], len(hyps)=beam_size
        for i, hyp in enumerate(hyps):
            score = 0.0
            for j, w in enumerate(hyp[0]):
                score += decoder_out[i][j][w]

            # last decoder output token is `eos`, for laste decoder input token.
            score += decoder_out[i][len(hyp[0])][self.model.eos]
            # add ctc score (which in ln domain)
            score += hyp[1] * self.ctc_decode_config.ctc_weight

            if score > best_score:
                best_score = score
                best_index = i

        # update the one best result
        # hyps stored the beam results and each fields is:

        logger.info(f"best hyp index: {best_index}")
        # logger.info(f'best result: {hyps[best_index]}')
        # the field of the hyps is:
        ## asr results
        # hyps[0][0]: the sentence word-id in the vocab with a tuple
        # hyps[0][1]: the sentence decoding probability with all paths
        ## timestamp
        # hyps[0][2]: viterbi_blank ending probability
        # hyps[0][3]: viterbi_non_blank dending probability
        # hyps[0][4]: current_token_prob,
        # hyps[0][5]: times_viterbi_blank ending timestamp,
        # hyps[0][6]: times_titerbi_non_blank encding timestamp.
        self.hyps = [hyps[best_index][0]]
        logger.info(f"best hyp ids: {self.hyps}")

        # update the hyps time stamp
        self.time_stamp = hyps[best_index][5] if hyps[best_index][2] > hyps[
            best_index][3] else hyps[best_index][6]
        logger.info(f"time stamp: {self.time_stamp}")

        # update one best result
        self.update_result()

        # update each word start and end time stamp
        # decoding frame to audio frame
        decode_frame_shift = self.model.encoder.embed.subsampling_rate
        decode_frame_shift_in_sec = decode_frame_shift * (self.n_shift /
                                                          self.sample_rate)
        logger.info(f"decode frame shift in sec: {decode_frame_shift_in_sec}")

        global_offset_in_sec = self.global_frame_offset * self.frame_shift_in_ms / 1000.0
        logger.info(f"global offset: {global_offset_in_sec} sec.")

        word_time_stamp = []
        for idx, _ in enumerate(self.time_stamp):
            start = (self.time_stamp[idx - 1] + self.time_stamp[idx]
                     ) / 2.0 if idx > 0 else 0
            start = start * decode_frame_shift_in_sec

            end = (self.time_stamp[idx] + self.time_stamp[idx + 1]
                   ) / 2.0 if idx < len(self.time_stamp) - 1 else self.offset

            end = end * decode_frame_shift_in_sec
            word_time_stamp.append({
                "w": self.result_transcripts[0][idx],
                "bg": global_offset_in_sec + start,
                "ed": global_offset_in_sec + end
            })

        self.word_time_stamp = word_time_stamp
        logger.info(f"word time stamp: {self.word_time_stamp}")


class ASRServerExecutor(ASRExecutor):
    def __init__(self):
        super().__init__()
        self.task_resource = CommonTaskResource(
            task='asr', model_format='dynamic', inference_mode='online')

    def update_config(self) -> None:
        if "deepspeech2" in self.model_type:
            with UpdateConfig(self.config):
                # download lm
                self.config.decode.lang_model_path = os.path.join(
                    MODEL_HOME, 'language_model',
                    self.config.decode.lang_model_path)

            lm_url = self.task_resource.res_dict['lm_url']
            lm_md5 = self.task_resource.res_dict['lm_md5']
            logger.debug(f"Start to load language model {lm_url}")
            self.download_lm(
                lm_url,
                os.path.dirname(self.config.decode.lang_model_path), lm_md5)
        elif "conformer" in self.model_type or "transformer" in self.model_type:
            with UpdateConfig(self.config):
                logger.debug("start to create the stream conformer asr engine")
                # update the decoding method
                if self.decode_method:
                    self.config.decode.decoding_method = self.decode_method
                # update num_decoding_left_chunks
                if self.num_decoding_left_chunks:
                    assert self.num_decoding_left_chunks == -1 or self.num_decoding_left_chunks >= 0, "num_decoding_left_chunks should be -1 or >=0"
                    self.config.decode.num_decoding_left_chunks = self.num_decoding_left_chunks
                # we only support ctc_prefix_beam_search and attention_rescoring dedoding method
                # Generally we set the decoding_method to attention_rescoring
                if self.config.decode.decoding_method not in [
                        "ctc_prefix_beam_search", "attention_rescoring"
                ]:
                    logger.debug(
                        "we set the decoding_method to attention_rescoring")
                    self.config.decode.decoding_method = "attention_rescoring"

                assert self.config.decode.decoding_method in [
                    "ctc_prefix_beam_search", "attention_rescoring"
                ], f"we only support ctc_prefix_beam_search and attention_rescoring dedoding method, current decoding method is {self.config.decode.decoding_method}"
        else:
            raise Exception(f"not support: {self.model_type}")

    def init_model(self) -> None:
        if "deepspeech2" in self.model_type:
            # AM predictor
            logger.debug("ASR engine start to init the am predictor")
            self.am_predictor = init_predictor(
                model_file=self.am_model,
                params_file=self.am_params,
                predictor_conf=self.am_predictor_conf)
        elif "conformer" in self.model_type or "transformer" in self.model_type:
            # load model
            # model_type: {model_name}_{dataset}
            model_name = self.model_type[:self.model_type.rindex('_')]
            logger.debug(f"model name: {model_name}")
            model_class = self.task_resource.get_model_class(model_name)
            model = model_class.from_config(self.config)
            self.model = model
            self.model.set_state_dict(paddle.load(self.am_model))
            self.model.eval()
        else:
            raise Exception(f"not support: {self.model_type}")

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

        self.model_type = model_type
        self.sample_rate = sample_rate
        self.decode_method = decode_method
        self.num_decoding_left_chunks = num_decoding_left_chunks
        # conf for paddleinference predictor or onnx
        self.am_predictor_conf = am_predictor_conf
        logger.debug(f"model_type: {self.model_type}")

        sample_rate_str = '16k' if sample_rate == 16000 else '8k'
        tag = model_type + '-' + lang + '-' + sample_rate_str
        self.task_resource.set_task_model(model_tag=tag)

        if cfg_path is None or am_model is None or am_params is None:
            self.res_path = self.task_resource.res_dir
            self.cfg_path = os.path.join(
                self.res_path, self.task_resource.res_dict['cfg_path'])

            self.am_model = os.path.join(self.res_path,
                                         self.task_resource.res_dict['model'])
            self.am_params = os.path.join(self.res_path,
                                          self.task_resource.res_dict['params'])
        else:
            self.cfg_path = os.path.abspath(cfg_path)
            self.am_model = os.path.abspath(am_model)
            self.am_params = os.path.abspath(am_params)
            self.res_path = os.path.dirname(
                os.path.dirname(os.path.abspath(self.cfg_path)))

        logger.debug("Load the pretrained model:")
        logger.debug(f"  tag = {tag}")
        logger.debug(f"  res_path: {self.res_path}")
        logger.debug(f"  cfg path: {self.cfg_path}")
        logger.debug(f"  am_model path: {self.am_model}")
        logger.debug(f"  am_params path: {self.am_params}")

        #Init body.
        self.config = CfgNode(new_allowed=True)
        self.config.merge_from_file(self.cfg_path)

        if self.config.spm_model_prefix:
            self.config.spm_model_prefix = os.path.join(
                self.res_path, self.config.spm_model_prefix)
            logger.debug(f"spm model path: {self.config.spm_model_prefix}")

        self.vocab = self.config.vocab_filepath

        self.text_feature = TextFeaturizer(
            unit_type=self.config.unit_type,
            vocab=self.config.vocab_filepath,
            spm_model_prefix=self.config.spm_model_prefix)

        self.update_config()

        # AM predictor
        self.init_model()

        logger.debug(f"create the {model_type} model success")
        return True


class ASREngine(BaseEngine):
    """ASR server resource

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

        logger.debug(f"paddlespeech_server set the device: {self.device}")

        if not self.init_model():
            logger.error(
                "Init the ASR server occurs error, please check the server configuration yaml"
            )
            return False

        logger.info("Initialize ASR server engine successfully on device: %s." %
                    (self.device))

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
