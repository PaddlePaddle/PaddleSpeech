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
import paddle
import soundfile
from yacs.config import CfgNode
import numpy as np

from ..executor import BaseExecutor
from ..utils import cli_register
from ..utils import download_and_decompress
from ..utils import logger
from ..utils import MODEL_HOME
from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.transform.transformation import Transformation
from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.s2t.utils.utility import UpdateConfig

__all__ = ['ASRExecutor']

pretrained_models = {
    "wenetspeech_zh_16k": {
        'url':
        'https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/conformer.model.tar.gz',
        'md5':
        '54e7a558a6e020c2f5fb224874943f97',
        'cfg_path':
        'conf/conformer.yaml',
        'ckpt_path':
        'exp/conformer/checkpoints/wenetspeech',
    }
}

model_alias = {
    "ds2_offline": "paddlespeech.s2t.models.ds2:DeepSpeech2Model",
    "ds2_online": "paddlespeech.s2t.models.ds2_online:DeepSpeech2ModelOnline",
    "conformer": "paddlespeech.s2t.models.u2:U2Model",
    "transformer": "paddlespeech.s2t.models.u2:U2Model",
    "wenetspeech": "paddlespeech.s2t.models.u2:U2Model",
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
            default='wenetspeech',
            help='Choose model type of asr task.')
        self.parser.add_argument(
            '--lang',
            type=str,
            default='zh',
            help='Choose model language. zh or en')
        self.parser.add_argument(
            "--sr",
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
            '--ckpt_path',
            type=str,
            default=None,
            help='Checkpoint file of model.')
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
                        ckpt_path: Optional[os.PathLike]=None
                        ):
        """
            Init model and other resources from a specific path.
        """
        if cfg_path is None or ckpt_path is None:
            sample_rate_str = '16k' if sample_rate == 16000 else '8k'
            tag = model_type + '_' + lang + '_' + sample_rate_str
            res_path = self._get_pretrained_path(tag)  # wenetspeech_zh
            self.cfg_path = os.path.join(res_path,
                                         pretrained_models[tag]['cfg_path'])
            self.ckpt_path = os.path.join(res_path,
                                          pretrained_models[tag]['ckpt_path'] + ".pdparams")
            logger.info(res_path)
            logger.info(self.cfg_path)
            logger.info(self.ckpt_path)
        else:
            self.cfg_path = os.path.abspath(cfg_path)
            self.ckpt_path = os.path.abspath(ckpt_path + ".pdparams")
            res_path = os.path.dirname(
                os.path.dirname(os.path.abspath(self.cfg_path)))

        #Init body.
        self.config = CfgNode(new_allowed=True)
        self.config.merge_from_file(self.cfg_path)
        self.config.decoding.decoding_method = "attention_rescoring"
        model_conf = self.config.model
        logger.info(model_conf)

        with UpdateConfig(model_conf):
            if model_type == "ds2_online" or model_type == "ds2_offline":
                from paddlespeech.s2t.io.collator import SpeechCollator
                self.config.collator.vocab_filepath = os.path.join(
                    res_path, self.config.collator.vocab_filepath)
                self.config.collator.mean_std_filepath = os.path.join(
                    res_path, self.config.collator.cmvn_path)
                self.collate_fn_test = SpeechCollator.from_config(self.config)
                text_feature = TextFeaturizer(
                    unit_type=self.config.collator.unit_type,
                    vocab_filepath=self.config.collator.vocab_filepath,
                    spm_model_prefix=self.config.collator.spm_model_prefix)
                model_conf.input_dim = self.collate_fn_test.feature_size
                model_conf.output_dim = text_feature.vocab_size
            elif model_type == "conformer" or model_type == "transformer" or model_type == "wenetspeech":
                self.config.collator.vocab_filepath = os.path.join(
                    res_path, self.config.collator.vocab_filepath)
                text_feature = TextFeaturizer(
                    unit_type=self.config.collator.unit_type,
                    vocab_filepath=self.config.collator.vocab_filepath,
                    spm_model_prefix=self.config.collator.spm_model_prefix)
                model_conf.input_dim = self.config.collator.feat_dim
                model_conf.output_dim = text_feature.vocab_size
            else:
                raise Exception("wrong type")
        self.config.freeze()
        # Enter the path of model root
        os.chdir(res_path)

        model_class = dynamic_import(model_type, model_alias)
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
        if model_type == "ds2_online" or model_type == "ds2_offline":
            audio, _ = self.collate_fn_test.process_utterance(
                audio_file=audio_file, transcript=" ")
            audio_len = audio.shape[0]
            audio = paddle.to_tensor(audio, dtype='float32')
            audio_len = paddle.to_tensor(audio_len)
            audio = paddle.unsqueeze(audio, axis=0)
            vocab_list = collate_fn_test.vocab_list
            self._inputs["audio"] = audio
            self._inputs["audio_len"] = audio_len
            logger.info(f"audio feat shape: {audio.shape}")

        elif model_type == "conformer" or model_type == "transformer" or model_type == "wenetspeech":
            logger.info("get the preprocess conf")
            preprocess_conf = os.path.join(
                os.path.dirname(os.path.abspath(self.cfg_path)),
                "preprocess.yaml")

            logger.info(preprocess_conf)
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
            text_feature = TextFeaturizer(
                unit_type=self.config.collator.unit_type,
                vocab_filepath=self.config.collator.vocab_filepath,
                spm_model_prefix=self.config.collator.spm_model_prefix)
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
        text_feature = TextFeaturizer(
            unit_type=self.config.collator.unit_type,
            vocab_filepath=self.config.collator.vocab_filepath,
            spm_model_prefix=self.config.collator.spm_model_prefix)
        cfg = self.config.decoding
        audio = self._inputs["audio"]
        audio_len = self._inputs["audio_len"]
        if model_type == "ds2_online" or model_type == "ds2_offline":
            result_transcripts = self.model.decode(
                audio,
                audio_len,
                text_feature.vocab_list,
                decoding_method=cfg.decoding_method,
                lang_model_path=cfg.lang_model_path,
                beam_alpha=cfg.alpha,
                beam_beta=cfg.beta,
                beam_size=cfg.beam_size,
                cutoff_prob=cfg.cutoff_prob,
                cutoff_top_n=cfg.cutoff_top_n,
                num_processes=cfg.num_proc_bsearch)
            self._outputs["result"] = result_transcripts[0]

        elif model_type == "conformer" or model_type == "transformer" or model_type == "wenetspeech":
            result_transcripts = self.model.decode(
                audio,
                audio_len,
                text_feature=text_feature,
                decoding_method=cfg.decoding_method,
                lang_model_path=cfg.lang_model_path,
                beam_alpha=cfg.alpha,
                beam_beta=cfg.beta,
                beam_size=cfg.beam_size,
                cutoff_prob=cfg.cutoff_prob,
                cutoff_top_n=cfg.cutoff_top_n,
                num_processes=cfg.num_proc_bsearch,
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

    def _pcm16to32(self, audio):
        assert(audio.dtype == np.int16)
        audio = audio.astype("float32")
        bits = np.iinfo(np.int16).bits
        audio = audio / (2**(bits - 1))
        return audio

    def _pcm32to16(self, audio):
        assert(audio.dtype == np.float32)
        bits = np.iinfo(np.int16).bits
        audio = audio * (2**(bits - 1))
        audio = np.round(audio).astype("int16")
        return audio

    def _check(self, audio_file: str, sample_rate: int):
        self.sample_rate = sample_rate
        if self.sample_rate != 16000 and self.sample_rate != 8000:
            logger.error(
                "please input --sr 8000 or --sr 16000"
            )
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
            logger.error(str(e))
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
            logger.warning(
                "The sample rate of the input file is not {}.\n \
                            The program will resample the wav file to {}.\n \
                            If the result does not meet your expectations，\n \
                            Please input the 16k 16 bit 1 channel wav file. \
                        "
                .format(self.sample_rate, self.sample_rate))
            while (True):
                logger.info(
                    "Whether to change the sample rate and the channel. Y: change the sample. N: exit the prgream."
                )
                content = input("Input(Y/N):")
                if content.strip() == "Y" or content.strip(
                ) == "y" or content.strip() == "yes" or content.strip() == "Yes":
                    logger.info(
                        "change the sampele rate, channel to 16k and 1 channel")
                    break
                elif content.strip() == "N" or content.strip(
                ) == "n" or content.strip() == "no" or content.strip() == "No":
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
        sample_rate = parser_args.sr
        config = parser_args.config
        ckpt_path = parser_args.ckpt_path
        audio_file = parser_args.input
        device = parser_args.device

        try:
            res = self(model, lang, sample_rate, config, ckpt_path,
                       audio_file, device)
            logger.info('ASR Result: {}'.format(res))
            return True
        except Exception as e:
            print(e)
            return False

    def __call__(self, model, lang, sample_rate, config, ckpt_path,
                 audio_file, device):
        """
            Python API to call an executor.
        """
        audio_file = os.path.abspath(audio_file)
        self._check(audio_file, sample_rate)
        paddle.set_device(device)
        self._init_from_path(model, lang, sample_rate, config, ckpt_path)
        self.preprocess(model, audio_file)
        self.infer(model)
        res = self.postprocess()  # Retrieve result of asr.

        return res
