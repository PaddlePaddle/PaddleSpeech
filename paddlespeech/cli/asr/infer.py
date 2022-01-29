# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import argparse
import os
import sys
from typing import List
from typing import Optional
from typing import Union

import librosa
import numpy as np
import paddle
import soundfile
from yacs.config import CfgNode

from ..download import get_path_from_url
from ..executor import BaseExecutor
from ..log import logger
from ..utils import cli_register
from ..utils import download_and_decompress
from ..utils import MODEL_HOME
from ..utils import stats_wrapper
from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.transform.transformation import Transformation
from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.s2t.utils.utility import UpdateConfig

__all__ = ['ASRExecutor']

pretrained_models = {
    # The tags for pretrained_models should be "{model_name}[_{dataset}][-{lang}][-...]".
    # e.g. "conformer_wenetspeech-zh-16k" and "panns_cnn6-32k".
    # Command line and python api use "{model_name}[_{dataset}]" as --model, usage:
    # "paddlespeech asr --model conformer_wenetspeech --lang zh --sr 16000 --input ./input.wav"
    "conformer_wenetspeech-zh-16k": {
        'url':
        'https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr1_conformer_wenetspeech_ckpt_0.1.1.model.tar.gz',
        'md5':
        '76cb19ed857e6623856b7cd7ebbfeda4',
        'cfg_path':
        'model.yaml',
        'ckpt_path':
        'exp/conformer/checkpoints/wenetspeech',
    },
    "transformer_librispeech-en-16k": {
        'url':
        'https://paddlespeech.bj.bcebos.com/s2t/librispeech/asr1/asr1_transformer_librispeech_ckpt_0.1.1.model.tar.gz',
        'md5':
        '2c667da24922aad391eacafe37bc1660',
        'cfg_path':
        'model.yaml',
        'ckpt_path':
        'exp/transformer/checkpoints/avg_10',
    },
    "deepspeech2offline_aishell-zh-16k": {
        'url':
        'https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_aishell_ckpt_0.1.1.model.tar.gz',
        'md5':
        '932c3593d62fe5c741b59b31318aa314',
        'cfg_path':
        'model.yaml',
        'ckpt_path':
        'exp/deepspeech2/checkpoints/avg_1',
        'lm_url':
        'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm',
        'lm_md5':
        '29e02312deb2e59b3c8686c7966d4fe3'
    },
    "deepspeech2online_aishell-zh-16k": {
        'url':
        'https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_online_aishell_ckpt_0.1.1.model.tar.gz',
        'md5':
        'd5e076217cf60486519f72c217d21b9b',
        'cfg_path':
        'model.yaml',
        'ckpt_path':
        'exp/deepspeech2_online/checkpoints/avg_1',
        'lm_url':
        'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm',
        'lm_md5':
        '29e02312deb2e59b3c8686c7966d4fe3'
    },
    "deepspeech2offline_librispeech-en-16k": {
        'url':
        'https://paddlespeech.bj.bcebos.com/s2t/librispeech/asr0/asr0_deepspeech2_librispeech_ckpt_0.1.1.model.tar.gz',
        'md5':
        'f5666c81ad015c8de03aac2bc92e5762',
        'cfg_path':
        'model.yaml',
        'ckpt_path':
        'exp/deepspeech2/checkpoints/avg_1',
        'lm_url':
        'https://deepspeech.bj.bcebos.com/en_lm/common_crawl_00.prune01111.trie.klm',
        'lm_md5':
        '099a601759d467cd0a8523ff939819c5'
    },
}

model_alias = {
    "deepspeech2offline":
    "paddlespeech.s2t.models.ds2:DeepSpeech2Model",
    "deepspeech2online":
    "paddlespeech.s2t.models.ds2_online:DeepSpeech2ModelOnline",
    "conformer":
    "paddlespeech.s2t.models.u2:U2Model",
    "transformer":
    "paddlespeech.s2t.models.u2:U2Model",
    "wenetspeech":
    "paddlespeech.s2t.models.u2:U2Model",
}


@cli_register(
    name='paddlespeech.asr', description='Speech to text infer command.')
class ASRExecutor(BaseExecutor):
    def __init__(self):
        super(ASRExecutor, self).__init__()

        self.parser = argparse.ArgumentParser(
            prog='paddlespeech.asr', add_help=True)
        self.parser.add_argument(
            '--input', type=str, required=True, help='Audio file to recognize.')
        self.parser.add_argument(
            '--model',
            type=str,
            default='conformer_wenetspeech',
            choices=[tag[:tag.index('-')] for tag in pretrained_models.keys()],
            help='Choose model type of asr task.')
        self.parser.add_argument(
            '--lang',
            type=str,
            default='zh',
            help='Choose model language. zh or en, zh:[conformer_wenetspeech-zh-16k], en:[transformer_librispeech-en-16k]'
        )
        self.parser.add_argument(
            "--sample_rate",
            type=int,
            default=16000,
            choices=[8000, 16000],
            help='Choose the audio sample rate of the model. 8000 or 16000')
        self.parser.add_argument(
            '--config',
            type=str,
            default=None,
            help='Config of asr task. Use deault config when it is None.')
        self.parser.add_argument(
            '--decode_method',
            type=str,
            default='attention_rescoring',
            choices=[
                'ctc_greedy_search', 'ctc_prefix_beam_search', 'attention',
                'attention_rescoring'
            ],
            help='only support transformer and conformer model')
        self.parser.add_argument(
            '--ckpt_path',
            type=str,
            default=None,
            help='Checkpoint file of model.')
        self.parser.add_argument(
            '--yes',
            '-y',
            action="store_true",
            default=False,
            help='No additional parameters required. Once set this parameter, it means accepting the request of the program by default, which includes transforming the audio sample rate'
        )
        self.parser.add_argument(
            '--device',
            type=str,
            default=paddle.get_device(),
            help='Choose device to execute model inference.')

    def _get_pretrained_path(self, tag: str) -> os.PathLike:
        """
        Download and returns pretrained resources path of current task.
        """
        assert tag in pretrained_models, 'Can not find pretrained resources of {}.'.format(
            tag)

        res_path = os.path.join(MODEL_HOME, tag)
        decompressed_path = download_and_decompress(pretrained_models[tag],
                                                    res_path)
        decompressed_path = os.path.abspath(decompressed_path)
        logger.info(
            'Use pretrained model stored in: {}'.format(decompressed_path))

        return decompressed_path

    def _init_from_path(self,
                        model_type: str='wenetspeech',
                        lang: str='zh',
                        sample_rate: int=16000,
                        cfg_path: Optional[os.PathLike]=None,
                        decode_method: str='attention_rescoring',
                        ckpt_path: Optional[os.PathLike]=None):
        """
        Init model and other resources from a specific path.
        """
        if hasattr(self, 'model'):
            logger.info('Model had been initialized.')
            return

        if cfg_path is None or ckpt_path is None:
            sample_rate_str = '16k' if sample_rate == 16000 else '8k'
            tag = model_type + '-' + lang + '-' + sample_rate_str
            res_path = self._get_pretrained_path(tag)  # wenetspeech_zh
            self.res_path = res_path
            self.cfg_path = os.path.join(res_path,
                                         pretrained_models[tag]['cfg_path'])
            self.ckpt_path = os.path.join(
                res_path, pretrained_models[tag]['ckpt_path'] + ".pdparams")
            logger.info(res_path)
            logger.info(self.cfg_path)
            logger.info(self.ckpt_path)
        else:
            self.cfg_path = os.path.abspath(cfg_path)
            self.ckpt_path = os.path.abspath(ckpt_path + ".pdparams")
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
                self.config.spm_model_prefix = os.path.join(
                    self.res_path, self.config.spm_model_prefix)
                self.text_feature = TextFeaturizer(
                    unit_type=self.config.unit_type,
                    vocab=self.config.vocab_filepath,
                    spm_model_prefix=self.config.spm_model_prefix)
                self.config.decode.decoding_method = decode_method

            else:
                raise Exception("wrong type")
        model_name = model_type[:model_type.rindex(
            '_')]  # model_type: {model_name}_{dataset}
        model_class = dynamic_import(model_name, model_alias)
        model_conf = self.config
        model = model_class.from_config(model_conf)
        self.model = model
        self.model.eval()

        # load model
        model_dict = paddle.load(self.ckpt_path)
        self.model.set_state_dict(model_dict)

    def preprocess(self, model_type: str, input: Union[str, os.PathLike]):
        """
        Input preprocess and return paddle.Tensor stored in self.input.
        Input content can be a text(tts), a file(asr, cls) or a streaming(not supported yet).
        """

        audio_file = input
        logger.info("Preprocess audio_file:" + audio_file)

        # Get the object for feature extraction
        if "deepspeech2online" in model_type or "deepspeech2offline" in model_type:
            audio, _ = self.collate_fn_test.process_utterance(
                audio_file=audio_file, transcript=" ")
            audio_len = audio.shape[0]
            audio = paddle.to_tensor(audio, dtype='float32')
            audio_len = paddle.to_tensor(audio_len)
            audio = paddle.unsqueeze(audio, axis=0)
            # vocab_list = collate_fn_test.vocab_list
            self._inputs["audio"] = audio
            self._inputs["audio_len"] = audio_len
            logger.info(f"audio feat shape: {audio.shape}")

        elif "conformer" in model_type or "transformer" in model_type or "wenetspeech" in model_type:
            logger.info("get the preprocess conf")
            preprocess_conf = self.config.preprocess_config
            preprocess_args = {"train": False}
            preprocessing = Transformation(preprocess_conf)
            logger.info("read the audio file")
            audio, audio_sample_rate = soundfile.read(
                audio_file, dtype="int16", always_2d=True)

            if self.change_format:
                if audio.shape[1] >= 2:
                    audio = audio.mean(axis=1, dtype=np.int16)
                else:
                    audio = audio[:, 0]
                # pcm16 -> pcm 32
                audio = self._pcm16to32(audio)
                audio = librosa.resample(audio, audio_sample_rate,
                                         self.sample_rate)
                audio_sample_rate = self.sample_rate
                # pcm32 -> pcm 16
                audio = self._pcm32to16(audio)
            else:
                audio = audio[:, 0]

            logger.info(f"audio shape: {audio.shape}")
            # fbank
            audio = preprocessing(audio, **preprocess_args)

            audio_len = paddle.to_tensor(audio.shape[0])
            audio = paddle.to_tensor(audio, dtype='float32').unsqueeze(axis=0)

            self._inputs["audio"] = audio
            self._inputs["audio_len"] = audio_len
            logger.info(f"audio feat shape: {audio.shape}")

        else:
            raise Exception("wrong type")

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
            self.model.decoder.init_decoder(
                decode_batch_size, self.text_feature.vocab_list,
                cfg.decoding_method, cfg.lang_model_path, cfg.alpha, cfg.beta,
                cfg.beam_size, cfg.cutoff_prob, cfg.cutoff_top_n,
                cfg.num_proc_bsearch)

            result_transcripts = self.model.decode(audio, audio_len)
            self.model.decoder.del_decoder()
            self._outputs["result"] = result_transcripts[0]

        elif "conformer" in model_type or "transformer" in model_type:
            result_transcripts = self.model.decode(
                audio,
                audio_len,
                text_feature=self.text_feature,
                decoding_method=cfg.decoding_method,
                beam_size=cfg.beam_size,
                ctc_weight=cfg.ctc_weight,
                decoding_chunk_size=cfg.decoding_chunk_size,
                num_decoding_left_chunks=cfg.num_decoding_left_chunks,
                simulate_streaming=cfg.simulate_streaming)
            self._outputs["result"] = result_transcripts[0][0]
        else:
            raise Exception("invalid model name")

    def postprocess(self) -> Union[str, os.PathLike]:
        """
            Output postprocess and return human-readable results such as texts and audio files.
        """
        return self._outputs["result"]

    def download_lm(self, url, lm_dir, md5sum):
        download_path = get_path_from_url(
            url=url,
            root_dir=lm_dir,
            md5sum=md5sum,
            decompress=False, )

    def _pcm16to32(self, audio):
        assert (audio.dtype == np.int16)
        audio = audio.astype("float32")
        bits = np.iinfo(np.int16).bits
        audio = audio / (2**(bits - 1))
        return audio

    def _pcm32to16(self, audio):
        assert (audio.dtype == np.float32)
        bits = np.iinfo(np.int16).bits
        audio = audio * (2**(bits - 1))
        audio = np.round(audio).astype("int16")
        return audio

    def _check(self, audio_file: str, sample_rate: int, force_yes: bool):
        self.sample_rate = sample_rate
        if self.sample_rate != 16000 and self.sample_rate != 8000:
            logger.error("please input --sr 8000 or --sr 16000")
            raise Exception("invalid sample rate")
            sys.exit(-1)

        if not os.path.isfile(audio_file):
            logger.error("Please input the right audio file path")
            sys.exit(-1)

        logger.info("checking the audio file format......")
        try:
            audio, audio_sample_rate = soundfile.read(
                audio_file, dtype="int16", always_2d=True)
        except Exception as e:
            logger.exception(e)
            logger.error(
                "can not open the audio file, please check the audio file format is 'wav'. \n \
                 you can try to use sox to change the file format.\n \
                 For example: \n \
                 sample rate: 16k \n \
                 sox input_audio.xx --rate 16k --bits 16 --channels 1 output_audio.wav \n \
                 sample rate: 8k \n \
                 sox input_audio.xx --rate 8k --bits 16 --channels 1 output_audio.wav \n \
                 ")
            sys.exit(-1)
        logger.info("The sample rate is %d" % audio_sample_rate)
        if audio_sample_rate != self.sample_rate:
            logger.warning("The sample rate of the input file is not {}.\n \
                            The program will resample the wav file to {}.\n \
                            If the result does not meet your expectations，\n \
                            Please input the 16k 16 bit 1 channel wav file. \
                        ".format(self.sample_rate, self.sample_rate))
            if force_yes is False:
                while (True):
                    logger.info(
                        "Whether to change the sample rate and the channel. Y: change the sample. N: exit the prgream."
                    )
                    content = input("Input(Y/N):")
                    if content.strip() == "Y" or content.strip(
                    ) == "y" or content.strip() == "yes" or content.strip(
                    ) == "Yes":
                        logger.info(
                            "change the sampele rate, channel to 16k and 1 channel"
                        )
                        break
                    elif content.strip() == "N" or content.strip(
                    ) == "n" or content.strip() == "no" or content.strip(
                    ) == "No":
                        logger.info("Exit the program")
                        exit(1)
                    else:
                        logger.warning("Not regular input, please input again")

            self.change_format = True
        else:
            logger.info("The audio file format is right")
            self.change_format = False

    def execute(self, argv: List[str]) -> bool:
        """
            Command line entry.
        """
        parser_args = self.parser.parse_args(argv)

        model = parser_args.model
        lang = parser_args.lang
        sample_rate = parser_args.sample_rate
        config = parser_args.config
        ckpt_path = parser_args.ckpt_path
        audio_file = parser_args.input
        decode_method = parser_args.decode_method
        force_yes = parser_args.yes
        device = parser_args.device

        try:
            res = self(audio_file, model, lang, sample_rate, config, ckpt_path,
                       decode_method, force_yes, device)
            logger.info('ASR Result: {}'.format(res))
            return True
        except Exception as e:
            logger.exception(e)
            return False

    @stats_wrapper
    def __call__(self,
                 audio_file: os.PathLike,
                 model: str='conformer_wenetspeech',
                 lang: str='zh',
                 sample_rate: int=16000,
                 config: os.PathLike=None,
                 ckpt_path: os.PathLike=None,
                 decode_method: str='attention_rescoring',
                 force_yes: bool=False,
                 device=paddle.get_device()):
        """
        Python API to call an executor.
        """
        audio_file = os.path.abspath(audio_file)
        self._check(audio_file, sample_rate, force_yes)
        paddle.set_device(device)
        self._init_from_path(model, lang, sample_rate, config, decode_method,
                             ckpt_path)
        self.preprocess(model, audio_file)
        self.infer(model)
        res = self.postprocess()  # Retrieve result of asr.

        return res
