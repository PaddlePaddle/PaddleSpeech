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
import copy
import os
from typing import Optional

import numpy as np
import paddle
from numpy import float32
from yacs.config import CfgNode

from .pretrained_models import pretrained_models
from paddlespeech.cli.asr.infer import ASRExecutor
from paddlespeech.cli.log import logger
from paddlespeech.cli.utils import MODEL_HOME
from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.frontend.speech import SpeechSegment
from paddlespeech.s2t.modules.ctc import CTCDecoder
from paddlespeech.s2t.transform.transformation import Transformation
from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.s2t.utils.tensor_utils import add_sos_eos
from paddlespeech.s2t.utils.tensor_utils import pad_sequence
from paddlespeech.s2t.utils.utility import UpdateConfig
from paddlespeech.server.engine.asr.online.ctc_search import CTCPrefixBeamSearch
from paddlespeech.server.engine.base_engine import BaseEngine
from paddlespeech.server.utils.audio_process import pcm2float
from paddlespeech.server.utils.paddle_predictor import init_predictor

__all__ = ['ASREngine']


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
        self.config = asr_engine.config
        self.model_config = asr_engine.executor.config
        self.asr_engine = asr_engine

        self.init()
        self.reset()

    def init(self):
        # model_type, sample_rate and text_feature is shared for deepspeech2 and conformer
        self.model_type = self.asr_engine.executor.model_type
        self.sample_rate = self.asr_engine.executor.sample_rate
        # tokens to text
        self.text_feature = self.asr_engine.executor.text_feature

        if "deepspeech2online" in self.model_type or "deepspeech2offline" in self.model_type:
            from paddlespeech.s2t.io.collator import SpeechCollator
            self.am_predictor = self.asr_engine.executor.am_predictor

            self.collate_fn_test = SpeechCollator.from_config(self.model_config)
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
            # frame window samples length and frame shift samples length

            self.win_length = int(self.model_config.window_ms / 1000 *
                                  self.sample_rate)
            self.n_shift = int(self.model_config.stride_ms / 1000 *
                               self.sample_rate)

        elif "conformer" in self.model_type or "transformer" in self.model_type:
            # acoustic model
            self.model = self.asr_engine.executor.model

            # ctc decoding config
            self.ctc_decode_config = self.asr_engine.executor.config.decode
            self.searcher = CTCPrefixBeamSearch(self.ctc_decode_config)

            # extract feat, new only fbank in conformer model
            self.preprocess_conf = self.model_config.preprocess_config
            self.preprocess_args = {"train": False}
            self.preprocessing = Transformation(self.preprocess_conf)

            # frame window samples length and frame shift samples length
            self.win_length = self.preprocess_conf.process[0]['win_length']
            self.n_shift = self.preprocess_conf.process[0]['n_shift']

    def extract_feat(self, samples):

        # we compute the elapsed time of first char occuring 
        # and we record the start time at the first pcm sample arraving
        # if self.first_char_occur_elapsed is not None:
        #     self.first_char_occur_elapsed = time.time()

        if "deepspeech2online" in self.model_type:
            # self.reamined_wav stores all the samples, 
            # include the original remained_wav and this package samples
            samples = np.frombuffer(samples, dtype=np.int16)
            assert samples.ndim == 1

            # pcm16 -> pcm 32
            # pcm2float will change the orignal samples, 
            # so we shoule do pcm2float before concatenate
            samples = pcm2float(samples)

            if self.remained_wav is None:
                self.remained_wav = samples
            else:
                assert self.remained_wav.ndim == 1
                self.remained_wav = np.concatenate([self.remained_wav, samples])
            logger.info(
                f"The connection remain the audio samples: {self.remained_wav.shape}"
            )

            # read audio
            speech_segment = SpeechSegment.from_pcm(
                self.remained_wav, self.sample_rate, transcript=" ")
            # audio augment
            self.collate_fn_test.augmentation.transform_audio(speech_segment)

            # extract speech feature
            spectrum, transcript_part = self.collate_fn_test._speech_featurizer.featurize(
                speech_segment, self.collate_fn_test.keep_transcription_text)
            # CMVN spectrum
            if self.collate_fn_test._normalizer:
                spectrum = self.collate_fn_test._normalizer.apply(spectrum)

            # spectrum augment
            audio = self.collate_fn_test.augmentation.transform_feature(
                spectrum)

            audio_len = audio.shape[0]
            audio = paddle.to_tensor(audio, dtype='float32')
            # audio_len = paddle.to_tensor(audio_len)
            audio = paddle.unsqueeze(audio, axis=0)

            if self.cached_feat is None:
                self.cached_feat = audio
            else:
                assert (len(audio.shape) == 3)
                assert (len(self.cached_feat.shape) == 3)
                self.cached_feat = paddle.concat(
                    [self.cached_feat, audio], axis=1)

                # set the feat device
            if self.device is None:
                self.device = self.cached_feat.place

            self.num_frames += audio_len
            self.remained_wav = self.remained_wav[self.n_shift * audio_len:]

            logger.info(
                f"process the audio feature success, the connection feat shape: {self.cached_feat.shape}"
            )
            logger.info(
                f"After extract feat, the connection remain the audio samples: {self.remained_wav.shape}"
            )
        elif "conformer_online" in self.model_type:
            logger.info("Online ASR extract the feat")
            samples = np.frombuffer(samples, dtype=np.int16)
            assert samples.ndim == 1

            logger.info(f"This package receive {samples.shape[0]} pcm data")
            self.num_samples += samples.shape[0]

            # self.reamined_wav stores all the samples, 
            # include the original remained_wav and this package samples
            if self.remained_wav is None:
                self.remained_wav = samples
            else:
                assert self.remained_wav.ndim == 1
                self.remained_wav = np.concatenate([self.remained_wav, samples])
            logger.info(
                f"The connection remain the audio samples: {self.remained_wav.shape}"
            )
            if len(self.remained_wav) < self.win_length:
                return 0

            # fbank
            x_chunk = self.preprocessing(self.remained_wav,
                                         **self.preprocess_args)
            x_chunk = paddle.to_tensor(
                x_chunk, dtype="float32").unsqueeze(axis=0)
            if self.cached_feat is None:
                self.cached_feat = x_chunk
            else:
                assert (len(x_chunk.shape) == 3)
                assert (len(self.cached_feat.shape) == 3)
                self.cached_feat = paddle.concat(
                    [self.cached_feat, x_chunk], axis=1)

            # set the feat device
            if self.device is None:
                self.device = self.cached_feat.place

            num_frames = x_chunk.shape[1]
            self.num_frames += num_frames
            self.remained_wav = self.remained_wav[self.n_shift * num_frames:]

            logger.info(
                f"process the audio feature success, the connection feat shape: {self.cached_feat.shape}"
            )
            logger.info(
                f"After extract feat, the connection remain the audio samples: {self.remained_wav.shape}"
            )
            # logger.info(f"accumulate samples: {self.num_samples}")       

    def reset(self):
        if "deepspeech2online" in self.model_type or "deepspeech2offline" in self.model_type:
            # for deepspeech2 
            self.chunk_state_h_box = copy.deepcopy(
                self.asr_engine.executor.chunk_state_h_box)
            self.chunk_state_c_box = copy.deepcopy(
                self.asr_engine.executor.chunk_state_c_box)
            self.decoder.reset_decoder(batch_size=1)

        # for conformer online
        self.subsampling_cache = None
        self.elayers_output_cache = None
        self.conformer_cnn_cache = None
        self.encoder_out = None
        self.cached_feat = None
        self.remained_wav = None
        self.offset = 0
        self.num_samples = 0
        self.device = None
        self.hyps = []
        self.num_frames = 0
        self.chunk_num = 0
        self.global_frame_offset = 0
        self.result_transcripts = ['']
        self.word_time_stamp = []
        self.time_stamp = []
        self.first_char_occur_elapsed = None
        self.word_time_stamp = None

    def decode(self, is_finished=False):
        if "deepspeech2online" in self.model_type:
            # x_chunk 是特征数据
            decoding_chunk_size = 1  # decoding_chunk_size=1 in deepspeech2 model
            context = 7  # context=7 in deepspeech2 model
            subsampling = 4  # subsampling=4 in deepspeech2 model
            stride = subsampling * decoding_chunk_size
            cached_feature_num = context - subsampling
            # decoding window for model
            decoding_window = (decoding_chunk_size - 1) * subsampling + context

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

            for cur in range(0, num_frames - left_frames + 1, stride):
                end = min(cur + decoding_window, num_frames)
                # extract the audio
                x_chunk = self.cached_feat[:, cur:end, :].numpy()
                x_chunk_lens = np.array([x_chunk.shape[1]])
                trans_best = self.decode_one_chunk(x_chunk, x_chunk_lens)

            self.result_transcripts = [trans_best]

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
        logger.info("start to decoce one chunk with deepspeech2 model")
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
        logger.info(f"decode one best result: {trans_best[0]}")
        return trans_best[0]

    @paddle.no_grad()
    def advance_decoding(self, is_finished=False):
        logger.info("start to decode with advanced_decoding method")
        cfg = self.ctc_decode_config
        decoding_chunk_size = cfg.decoding_chunk_size
        num_decoding_left_chunks = cfg.num_decoding_left_chunks

        assert decoding_chunk_size > 0
        subsampling = self.model.encoder.embed.subsampling_rate
        context = self.model.encoder.embed.right_context + 1
        stride = subsampling * decoding_chunk_size
        cached_feature_num = context - subsampling  # processed chunk feature cached for next chunk

        # decoding window for model
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
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
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks
        outputs = []

        # num_frames - context + 1 ensure that current frame can get context window
        if is_finished:
            # if get the finished chunk, we need process the last context
            left_frames = context
        else:
            # we only process decoding_window frames for one chunk
            left_frames = decoding_window

        # record the end for removing the processed feat
        end = None
        for cur in range(0, num_frames - left_frames + 1, stride):
            end = min(cur + decoding_window, num_frames)

            self.chunk_num += 1
            chunk_xs = self.cached_feat[:, cur:end, :]
            (y, self.subsampling_cache, self.elayers_output_cache,
             self.conformer_cnn_cache) = self.model.encoder.forward_chunk(
                 chunk_xs, self.offset, required_cache_size,
                 self.subsampling_cache, self.elayers_output_cache,
                 self.conformer_cnn_cache)
            outputs.append(y)

            # update the offset
            self.offset += y.shape[1]

        ys = paddle.cat(outputs, 1)
        if self.encoder_out is None:
            self.encoder_out = ys
        else:
            self.encoder_out = paddle.concat([self.encoder_out, ys], axis=1)

        # get the ctc probs
        ctc_probs = self.model.ctc.log_softmax(ys)  # (1, maxlen, vocab_size)
        ctc_probs = ctc_probs.squeeze(0)

        self.searcher.search(ctc_probs, self.cached_feat.place)

        self.hyps = self.searcher.get_one_best_hyps()
        assert self.cached_feat.shape[0] == 1
        assert end >= cached_feature_num

        self.cached_feat = self.cached_feat[0, end -
                                            cached_feature_num:, :].unsqueeze(0)
        assert len(
            self.cached_feat.shape
        ) == 3, f"current cache feat shape is: {self.cached_feat.shape}"

        logger.info(
            f"This connection handler encoder out shape: {self.encoder_out.shape}"
        )

    def update_result(self):
        logger.info("update the final result")
        hyps = self.hyps
        self.result_transcripts = [
            self.text_feature.defeaturize(hyp) for hyp in hyps
        ]
        self.result_tokenids = [hyp for hyp in hyps]

    def get_result(self):
        if len(self.result_transcripts) > 0:
            return self.result_transcripts[0]
        else:
            return ''

    def get_word_time_stamp(self):
        return self.word_time_stamp

    @paddle.no_grad()
    def rescoring(self):
        if "deepspeech2online" in self.model_type or "deepspeech2offline" in self.model_type:
            return

        logger.info("rescoring the final result")
        if "attention_rescoring" != self.ctc_decode_config.decoding_method:
            return

        self.searcher.finalize_search()
        self.update_result()

        beam_size = self.ctc_decode_config.beam_size
        hyps = self.searcher.get_hyps()
        if hyps is None or len(hyps) == 0:
            return

        # assert len(hyps) == beam_size
        hyp_list = []
        for hyp in hyps:
            hyp_content = hyp[0]
            # Prevent the hyp is empty
            if len(hyp_content) == 0:
                hyp_content = (self.model.ctc.blank_id, )
            hyp_content = paddle.to_tensor(
                hyp_content, place=self.device, dtype=paddle.long)
            hyp_list.append(hyp_content)
        hyps_pad = pad_sequence(hyp_list, True, self.model.ignore_id)
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

        logger.info(f"best index: {best_index}")
        # logger.info(f'best result: {hyps[best_index]}')
        # the field of the hyps is:
        # hyps[0][0]: the sentence word-id in the vocab with a tuple
        # hyps[0][1]: the sentence decoding probability with all paths
        # hyps[0][2]: viterbi_blank ending probability
        # hyps[0][3]: viterbi_non_blank probability
        # hyps[0][4]: current_token_prob,
        # hyps[0][5]: times_viterbi_blank, 
        # hyps[0][6]: times_titerbi_non_blank 
        self.hyps = [hyps[best_index][0]]

        # update the hyps time stamp
        self.time_stamp = hyps[best_index][5] if hyps[best_index][2] > hyps[
            best_index][3] else hyps[best_index][6]
        logger.info(f"time stamp: {self.time_stamp}")

        self.update_result()

        # update each word start and end time stamp
        frame_shift_in_ms = self.model.encoder.embed.subsampling_rate * self.n_shift / self.sample_rate
        logger.info(f"frame shift ms: {frame_shift_in_ms}")
        word_time_stamp = []
        for idx, _ in enumerate(self.time_stamp):
            start = (self.time_stamp[idx - 1] + self.time_stamp[idx]
                     ) / 2.0 if idx > 0 else 0
            start = start * frame_shift_in_ms

            end = (self.time_stamp[idx] + self.time_stamp[idx + 1]
                   ) / 2.0 if idx < len(self.time_stamp) - 1 else self.offset
            end = end * frame_shift_in_ms
            word_time_stamp.append({
                "w": self.result_transcripts[0][idx],
                "bg": start,
                "ed": end
            })
            # logger.info(f"{self.result_transcripts[0][idx]}, start: {start}, end: {end}")
        self.word_time_stamp = word_time_stamp
        logger.info(f"word time stamp: {self.word_time_stamp}")


class ASRServerExecutor(ASRExecutor):
    def __init__(self):
        super().__init__()
        self.pretrained_models = pretrained_models

    def _init_from_path(self,
                        model_type: str='deepspeech2online_aishell',
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
        self.model_type = model_type
        self.sample_rate = sample_rate
        sample_rate_str = '16k' if sample_rate == 16000 else '8k'
        tag = model_type + '-' + lang + '-' + sample_rate_str
        if cfg_path is None or am_model is None or am_params is None:
            logger.info(f"Load the pretrained model, tag = {tag}")
            res_path = self._get_pretrained_path(tag)  # wenetspeech_zh
            self.res_path = res_path

            self.cfg_path = os.path.join(
                res_path, self.pretrained_models[tag]['cfg_path'])

            self.am_model = os.path.join(res_path,
                                         self.pretrained_models[tag]['model'])
            self.am_params = os.path.join(res_path,
                                          self.pretrained_models[tag]['params'])
            logger.info(res_path)
        else:
            self.cfg_path = os.path.abspath(cfg_path)
            self.am_model = os.path.abspath(am_model)
            self.am_params = os.path.abspath(am_params)
            self.res_path = os.path.dirname(
                os.path.dirname(os.path.abspath(self.cfg_path)))

        logger.info(self.cfg_path)
        logger.info(self.am_model)
        logger.info(self.am_params)

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

                lm_url = self.pretrained_models[tag]['lm_url']
                lm_md5 = self.pretrained_models[tag]['lm_md5']
                logger.info(f"Start to load language model {lm_url}")
                self.download_lm(
                    lm_url,
                    os.path.dirname(self.config.decode.lang_model_path), lm_md5)
            elif "conformer" in model_type or "transformer" in model_type:
                logger.info("start to create the stream conformer asr engine")
                if self.config.spm_model_prefix:
                    self.config.spm_model_prefix = os.path.join(
                        self.res_path, self.config.spm_model_prefix)
                self.vocab = self.config.vocab_filepath
                self.text_feature = TextFeaturizer(
                    unit_type=self.config.unit_type,
                    vocab=self.config.vocab_filepath,
                    spm_model_prefix=self.config.spm_model_prefix)
                # update the decoding method
                if decode_method:
                    self.config.decode.decoding_method = decode_method

                # we only support ctc_prefix_beam_search and attention_rescoring dedoding method
                # Generally we set the decoding_method to attention_rescoring
                if self.config.decode.decoding_method not in [
                        "ctc_prefix_beam_search", "attention_rescoring"
                ]:
                    logger.info(
                        "we set the decoding_method to attention_rescoring")
                    self.config.decode.decoding = "attention_rescoring"
                assert self.config.decode.decoding_method in [
                    "ctc_prefix_beam_search", "attention_rescoring"
                ], f"we only support ctc_prefix_beam_search and attention_rescoring dedoding method, current decoding method is {self.config.decode.decoding_method}"
            else:
                raise Exception("wrong type")
        if "deepspeech2online" in model_type or "deepspeech2offline" in model_type:
            # AM predictor
            logger.info("ASR engine start to init the am predictor")
            self.am_predictor_conf = am_predictor_conf
            self.am_predictor = init_predictor(
                model_file=self.am_model,
                params_file=self.am_params,
                predictor_conf=self.am_predictor_conf)

            # decoder
            logger.info("ASR engine start to create the ctc decoder instance")
            self.decoder = CTCDecoder(
                odim=self.config.output_dim,  # <blank> is in  vocab
                enc_n_units=self.config.rnn_layer_size * 2,
                blank_id=self.config.blank_id,
                dropout_rate=0.0,
                reduction=True,  # sum
                batch_average=True,  # sum / batch_size
                grad_norm_type=self.config.get('ctc_grad_norm_type', None))

            # init decoder
            logger.info("ASR engine start to init the ctc decoder")
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
        elif "conformer" in model_type or "transformer" in model_type:
            model_name = model_type[:model_type.rindex(
                '_')]  # model_type: {model_name}_{dataset}
            logger.info(f"model name: {model_name}")
            model_class = dynamic_import(model_name, self.model_alias)
            model_conf = self.config
            model = model_class.from_config(model_conf)
            self.model = model
            self.model.eval()

            # load model
            model_dict = paddle.load(self.am_model)
            self.model.set_state_dict(model_dict)
            logger.info("create the transformer like model success")

            # update the ctc decoding
            self.searcher = CTCPrefixBeamSearch(self.config.decode)
            self.transformer_decode_reset()

    def reset_decoder_and_chunk(self):
        """reset decoder and chunk state for an new audio
        """
        if "deepspeech2online" in self.model_type or "deepspeech2offline" in self.model_type:
            self.decoder.reset_decoder(batch_size=1)
            # init state box, for new audio request
            self.chunk_state_h_box = np.zeros(
                (self.config.num_rnn_layers, 1, self.config.rnn_layer_size),
                dtype=float32)
            self.chunk_state_c_box = np.zeros(
                (self.config.num_rnn_layers, 1, self.config.rnn_layer_size),
                dtype=float32)
        elif "conformer" in self.model_type or "transformer" in self.model_type:
            self.transformer_decode_reset()

    def decode_one_chunk(self, x_chunk, x_chunk_lens, model_type: str):
        """decode one chunk

        Args:
            x_chunk (numpy.array): shape[B, T, D]
            x_chunk_lens (numpy.array): shape[B]
            model_type (str): online model type

        Returns:
            str: one best result
        """
        logger.info("start to decoce chunk by chunk")
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
            logger.info(f"decode one best result: {trans_best[0]}")
            return trans_best[0]

        elif "conformer" in model_type or "transformer" in model_type:
            try:
                logger.info(
                    f"we will use the transformer like model : {self.model_type}"
                )
                self.advanced_decoding(x_chunk, x_chunk_lens)
                self.update_result()

                return self.result_transcripts[0]
            except Exception as e:
                logger.exception(e)
        else:
            raise Exception("invalid model name")

    def advanced_decoding(self, xs: paddle.Tensor, x_chunk_lens):
        logger.info("start to decode with advanced_decoding method")
        encoder_out, encoder_mask = self.encoder_forward(xs)
        ctc_probs = self.model.ctc.log_softmax(
            encoder_out)  # (1, maxlen, vocab_size)
        ctc_probs = ctc_probs.squeeze(0)
        self.searcher.search(ctc_probs, xs.place)
        # update the one best result
        self.hyps = self.searcher.get_one_best_hyps()

        # now we supprot ctc_prefix_beam_search and attention_rescoring
        if "attention_rescoring" in self.config.decode.decoding_method:
            self.rescoring(encoder_out, xs.place)

    def encoder_forward(self, xs):
        logger.info("get the model out from the feat")
        cfg = self.config.decode
        decoding_chunk_size = cfg.decoding_chunk_size
        num_decoding_left_chunks = cfg.num_decoding_left_chunks

        assert decoding_chunk_size > 0
        subsampling = self.model.encoder.embed.subsampling_rate
        context = self.model.encoder.embed.right_context + 1
        stride = subsampling * decoding_chunk_size

        # decoding window for model
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        num_frames = xs.shape[1]
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks

        logger.info("start to do model forward")
        outputs = []

        # num_frames - context + 1 ensure that current frame can get context window
        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + decoding_window, num_frames)
            chunk_xs = xs[:, cur:end, :]
            (y, self.subsampling_cache, self.elayers_output_cache,
             self.conformer_cnn_cache) = self.model.encoder.forward_chunk(
                 chunk_xs, self.offset, required_cache_size,
                 self.subsampling_cache, self.elayers_output_cache,
                 self.conformer_cnn_cache)
            outputs.append(y)
            self.offset += y.shape[1]

        ys = paddle.cat(outputs, 1)
        masks = paddle.ones([1, ys.shape[1]], dtype=paddle.bool)
        masks = masks.unsqueeze(1)
        return ys, masks

    def rescoring(self, encoder_out, device):
        logger.info("start to rescoring the hyps")
        beam_size = self.config.decode.beam_size
        hyps = self.searcher.get_hyps()
        assert len(hyps) == beam_size

        hyp_list = []
        for hyp in hyps:
            hyp_content = hyp[0]
            # Prevent the hyp is empty
            if len(hyp_content) == 0:
                hyp_content = (self.model.ctc.blank_id, )
            hyp_content = paddle.to_tensor(
                hyp_content, place=device, dtype=paddle.long)
            hyp_list.append(hyp_content)
        hyps_pad = pad_sequence(hyp_list, True, self.model.ignore_id)
        hyps_lens = paddle.to_tensor(
            [len(hyp[0]) for hyp in hyps], place=device,
            dtype=paddle.long)  # (beam_size,)
        hyps_pad, _ = add_sos_eos(hyps_pad, self.model.sos, self.model.eos,
                                  self.model.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining

        encoder_out = encoder_out.repeat(beam_size, 1, 1)
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
            score += hyp[1] * self.config.decode.ctc_weight
            if score > best_score:
                best_score = score
                best_index = i

        # update the one best result
        self.hyps = [hyps[best_index][0]]
        return hyps[best_index][0]

    def transformer_decode_reset(self):
        self.subsampling_cache = None
        self.elayers_output_cache = None
        self.conformer_cnn_cache = None
        self.offset = 0
        # decoding reset
        self.searcher.reset()

    def update_result(self):
        logger.info("update the final result")
        hyps = self.hyps
        self.result_transcripts = [
            self.text_feature.defeaturize(hyp) for hyp in hyps
        ]
        self.result_tokenids = [hyp for hyp in hyps]

    def extract_feat(self, samples, sample_rate):
        """extract feat

        Args:
            samples (numpy.array): numpy.float32
            sample_rate (int): sample rate

        Returns:
            x_chunk (numpy.array): shape[B, T, D]
            x_chunk_lens (numpy.array): shape[B]
        """

        if "deepspeech2online" in self.model_type:
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
            audio = self.collate_fn_test.augmentation.transform_feature(
                spectrum)

            audio_len = audio.shape[0]
            audio = paddle.to_tensor(audio, dtype='float32')
            # audio_len = paddle.to_tensor(audio_len)
            audio = paddle.unsqueeze(audio, axis=0)

            x_chunk = audio.numpy()
            x_chunk_lens = np.array([audio_len])

            return x_chunk, x_chunk_lens
        elif "conformer_online" in self.model_type:

            if sample_rate != self.sample_rate:
                logger.info(f"audio sample rate {sample_rate} is not match,"
                            "the model sample_rate is {self.sample_rate}")
            logger.info(f"ASR Engine use the {self.model_type} to process")
            logger.info("Create the preprocess instance")
            preprocess_conf = self.config.preprocess_config
            preprocess_args = {"train": False}
            preprocessing = Transformation(preprocess_conf)

            logger.info("Read the audio file")
            logger.info(f"audio shape: {samples.shape}")
            # fbank
            x_chunk = preprocessing(samples, **preprocess_args)
            x_chunk_lens = paddle.to_tensor(x_chunk.shape[0])
            x_chunk = paddle.to_tensor(
                x_chunk, dtype="float32").unsqueeze(axis=0)
            logger.info(
                f"process the audio feature success, feat shape: {x_chunk.shape}"
            )
            return x_chunk, x_chunk_lens


class ASREngine(BaseEngine):
    """ASR server engine

    Args:
        metaclass: Defaults to Singleton.
    """

    def __init__(self):
        super(ASREngine, self).__init__()
        logger.info("create the online asr engine instance")

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
        try:
            if self.config.get("device", None):
                self.device = self.config.device
            else:
                self.device = paddle.get_device()
            logger.info(f"paddlespeech_server set the device: {self.device}")
            paddle.set_device(self.device)
        except BaseException:
            logger.error(
                "Set device failed, please check if device is already used and the parameter 'device' in the yaml file"
            )

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

    def preprocess(self,
                   samples,
                   sample_rate,
                   model_type="deepspeech2online_aishell-zh-16k"):
        """preprocess

        Args:
            samples (numpy.array): numpy.float32
            sample_rate (int): sample rate

        Returns:
            x_chunk (numpy.array): shape[B, T, D]
            x_chunk_lens (numpy.array): shape[B]
        """
        # if "deepspeech" in model_type:
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
