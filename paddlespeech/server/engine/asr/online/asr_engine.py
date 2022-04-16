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
from paddlespeech.s2t.utils.utility import log_add
from typing import Optional
from collections import defaultdict
import numpy as np
import paddle
from numpy import float32
from yacs.config import CfgNode

from paddlespeech.cli.asr.infer import ASRExecutor
from paddlespeech.cli.asr.infer import model_alias
from paddlespeech.cli.asr.infer import pretrained_models
from paddlespeech.cli.log import logger
from paddlespeech.cli.utils import download_and_decompress
from paddlespeech.cli.utils import MODEL_HOME
from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.frontend.speech import SpeechSegment
from paddlespeech.s2t.modules.ctc import CTCDecoder
from paddlespeech.s2t.modules.mask import mask_finished_preds
from paddlespeech.s2t.modules.mask import mask_finished_scores
from paddlespeech.s2t.modules.mask import subsequent_mask
from paddlespeech.s2t.transform.transformation import Transformation
from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.s2t.utils.utility import UpdateConfig
from paddlespeech.server.engine.base_engine import BaseEngine
from paddlespeech.server.utils.audio_process import pcm2float
from paddlespeech.server.utils.paddle_predictor import init_predictor

__all__ = ['ASREngine']

pretrained_models = {
    "deepspeech2online_aishell-zh-16k": {
        'url':
        'https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_online_aishell_ckpt_0.2.0.model.tar.gz',
        'md5':
        '23e16c69730a1cb5d735c98c83c21e16',
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
    "conformer2online_aishell-zh-16k": {
        'url':
        'https://paddlespeech.bj.bcebos.com/s2t/multi_cn/asr1/asr1_chunk_conformer_multi_cn_ckpt_0.2.0.model.tar.gz',
        'md5':
        '7989b3248c898070904cf042fd656003',
        'cfg_path':
        'model.yaml',
        'ckpt_path':
        'exp/chunk_conformer/checkpoints/multi_cn',
        'model':
        'exp/chunk_conformer/checkpoints/multi_cn.pdparams',
        'params':
        'exp/chunk_conformer/checkpoints/multi_cn.pdparams',
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
        self.model_type = model_type
        self.sample_rate = sample_rate
        if cfg_path is None or am_model is None or am_params is None:
            sample_rate_str = '16k' if sample_rate == 16000 else '8k'
            tag = model_type + '-' + lang + '-' + sample_rate_str
            logger.info(f"Load the pretrained model, tag = {tag}")
            res_path = self._get_pretrained_path(tag)  # wenetspeech_zh
            self.res_path = res_path
            self.cfg_path = "/home/users/xiongxinlei/task/paddlespeech-develop/PaddleSpeech/examples/aishell/asr1/model.yaml"
            # self.cfg_path = os.path.join(res_path,
            #                              pretrained_models[tag]['cfg_path'])

            self.am_model = os.path.join(res_path,
                                         pretrained_models[tag]['model'])
            self.am_params = os.path.join(res_path,
                                          pretrained_models[tag]['params'])
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

                lm_url = pretrained_models[tag]['lm_url']
                lm_md5 = pretrained_models[tag]['lm_md5']
                logger.info(f"Start to load language model {lm_url}")
                self.download_lm(
                    lm_url,
                    os.path.dirname(self.config.decode.lang_model_path), lm_md5)
            elif "conformer" in model_type or "transformer" in model_type or "wenetspeech" in model_type:
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
        elif "conformer" in model_type or "transformer" in model_type or "wenetspeech" in model_type:
            model_name = model_type[:model_type.rindex(
                '_')]  # model_type: {model_name}_{dataset}
            logger.info(f"model name: {model_name}")
            model_class = dynamic_import(model_name, model_alias)
            model_conf = self.config
            model = model_class.from_config(model_conf)
            self.model = model
            self.model.eval()

            # load model
            model_dict = paddle.load(self.am_model)
            self.model.set_state_dict(model_dict)
            logger.info("create the transformer like model success")

            # update the ctc decoding
            self.searcher = None
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
        elif "conformer" in self.model_type or "transformer" in self.model_type or "wenetspeech" in self.model_type:
            self.transformer_decode_reset()

    def decode_one_chunk(self, x_chunk, x_chunk_lens, model_type: str):
        """decode one chunk

        Args:
            x_chunk (numpy.array): shape[B, T, D]
            x_chunk_lens (numpy.array): shape[B]
            model_type (str): online model type

        Returns:
            [type]: [description]
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
            logger.info(f"decode one one best result: {trans_best[0]}")
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
        encoder_out, encoder_mask = self.decode_forward(xs)
        self.ctc_prefix_beam_search(xs, encoder_out, encoder_mask)

    def decode_forward(self, xs):
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

    def transformer_decode_reset(self):
        self.subsampling_cache = None
        self.elayers_output_cache = None
        self.conformer_cnn_cache = None
        self.hyps = None
        self.offset = 0
        self.cur_hyps = None
        self.hyps = None

    def ctc_prefix_beam_search(self, xs, encoder_out, encoder_mask, blank_id=0):
        # decode 
        logger.info("start to ctc prefix search")

        device = xs.place
        cfg = self.config.decode
        batch_size = xs.shape[0]
        beam_size = cfg.beam_size
        maxlen = encoder_out.shape[1]

        ctc_probs = self.model.ctc.log_softmax(encoder_out)  # (1, maxlen, vocab_size)
        ctc_probs = ctc_probs.squeeze(0)
        
        # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
        # blank_ending_score and  none_blank_ending_score in ln domain
        if self.cur_hyps is None:
            self.cur_hyps = [(tuple(), (0.0, -float('inf')))]
        # 2. CTC beam search step by step
        for t in range(0, maxlen):
            logp = ctc_probs[t]  # (vocab_size,)
            # key: prefix, value (pb, pnb), default value(-inf, -inf)
            next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
            
            # 2.1 First beam prune: select topk best
            #     do token passing process
            top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
            for s in top_k_index:
                s = s.item()
                ps = logp[s].item()
                for prefix, (pb, pnb) in self.cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == blank_id:  # blank
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pb = log_add([n_pb, pb + ps, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                    elif s == last:
                        #  Update *ss -> *s;
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pnb = log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
                    else:
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)

            # 2.2 Second beam prune
            next_hyps = sorted(
                next_hyps.items(),
                key=lambda x: log_add(list(x[1])),
                reverse=True)
            self.cur_hyps = next_hyps[:beam_size]

        hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in self.cur_hyps]
        
        self.hyps = [hyps[0][0]]
        logger.info("ctc prefix search success")
        return hyps, encoder_out

    def update_result(self):
        logger.info("update the final result")
        self.result_transcripts = [
            self.text_feature.defeaturize(hyp) for hyp in self.hyps
        ]
        self.result_tokenids = [hyp for hyp in self.hyps]

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
        elif "conformer2online" in self.model_type:

            if sample_rate != self.sample_rate:
                logger.info(f"audio sample rate {sample_rate} is not match," \
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
        logger.info("create the online asr engine instache")

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
