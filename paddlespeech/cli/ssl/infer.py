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
import io
import os
import sys
import time
from collections import OrderedDict
from typing import List
from typing import Optional
from typing import Union

import librosa
import numpy as np
import paddle
import soundfile
from paddlenlp.transformers import AutoTokenizer
from yacs.config import CfgNode

from ..executor import BaseExecutor
from ..log import logger
from ..utils import CLI_TIMER
from ..utils import stats_wrapper
from ..utils import timer_register
from paddlespeech.audio.transform.transformation import Transformation
from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.utils.utility import UpdateConfig

__all__ = ['SSLExecutor']


@timer_register
class SSLExecutor(BaseExecutor):
    def __init__(self):
        super().__init__('ssl')
        self.parser = argparse.ArgumentParser(
            prog='paddlespeech.ssl', add_help=True)
        self.parser.add_argument(
            '--input', type=str, default=None, help='Audio file to recognize.')
        self.parser.add_argument(
            '--model',
            type=str,
            default='wav2vec2',
            choices=['wav2vec2', 'hubert', "wavlm"],
            help='Choose model type of asr task.')
        self.parser.add_argument(
            '--task',
            type=str,
            default='asr',
            choices=['asr', 'vector'],
            help='Choose output type for ssl task')
        self.parser.add_argument(
            '--lang',
            type=str,
            default='en',
            help='Choose model language. zh or en, zh:[wav2vec2ASR_aishell1-zh-16k], en:[wav2vec2ASR_librispeech-en-16k, hubertASR_librispeech_100-en-16k]'
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
            default='ctc_greedy_search',
            choices=[
                'ctc_greedy_search',
                'ctc_prefix_beam_search',
            ],
            help='only support asr task')
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
            help='No additional parameters required. \
            Once set this parameter, it means accepting the request of the program by default, \
            which includes transforming the audio sample rate')
        self.parser.add_argument(
            '--rtf',
            action="store_true",
            default=False,
            help='Show Real-time Factor(RTF).')
        self.parser.add_argument(
            '--device',
            type=str,
            default=paddle.get_device(),
            help='Choose device to execute model inference.')
        self.parser.add_argument(
            '-d',
            '--job_dump_result',
            action='store_true',
            help='Save job result into file.')
        self.parser.add_argument(
            '-v',
            '--verbose',
            action='store_true',
            help='Increase logger verbosity of current task.')
        self.last_call_params = None

    def _init_from_path(self,
                        model_type: str=None,
                        task: str='asr',
                        lang: str='en',
                        sample_rate: int=16000,
                        cfg_path: Optional[os.PathLike]=None,
                        decode_method: str='ctc_greedy_search',
                        ckpt_path: Optional[os.PathLike]=None):
        """
        Init model and other resources from a specific path.
        """
        logger.debug("start to init the model")

        if model_type is None:
            logger.debug(
                "Model type had not been specified, default {} was used.".
                format(model_type))
        # default max_len: unit:second
        self.max_len = 50
        if hasattr(self, 'model'):
            logger.debug('Model had been initialized.')
            return
        if cfg_path is None or ckpt_path is None:
            sample_rate_str = '16k' if sample_rate == 16000 else '8k'
            if task == 'asr':
                if model_type == 'wav2vec2':
                    if lang == 'en':
                        model_prefix = 'wav2vec2ASR_librispeech'
                    elif lang == 'zh':
                        model_prefix = 'wav2vec2ASR_aishell1'
                    tag = model_prefix + '-' + lang + '-' + sample_rate_str
                elif model_type == 'hubert':
                    if lang == 'en':
                        model_prefix = 'hubertASR_librispeech-100h'
                    elif lang == 'zh':
                        logger.error("zh hubertASR is not supported yet")
                    tag = model_prefix + '-' + lang + '-' + sample_rate_str
                elif model_type == 'wavlm':
                    if lang == "en":
                        model_prefix = "wavlmASR_librispeech"
                    elif lang == "zh":
                        logger.error("zh wavlmASR is not supported yet")
                    tag = model_prefix + '-' + lang + '-' + sample_rate_str
            else:
                tag = model_type + '-' + lang + '-' + sample_rate_str
            self.task_resource.set_task_model(tag, version=None)
            self.res_path = self.task_resource.res_dir

            self.cfg_path = os.path.join(
                self.res_path, self.task_resource.res_dict['cfg_path'])
            self.ckpt_path = os.path.join(
                self.res_path,
                self.task_resource.res_dict['ckpt_path'] + ".pdparams")
            logger.debug(self.res_path)
        else:
            self.cfg_path = os.path.abspath(cfg_path)
            self.ckpt_path = os.path.abspath(ckpt_path + ".pdparams")
            self.res_path = os.path.dirname(
                os.path.dirname(os.path.abspath(self.cfg_path)))
        logger.debug(self.cfg_path)
        logger.debug(self.ckpt_path)

        #Init body.
        self.config = CfgNode(new_allowed=True)
        self.config.merge_from_file(self.cfg_path)
        if task == 'asr':
            with UpdateConfig(self.config):
                if lang == 'en':
                    self.text_feature = TextFeaturizer(
                        unit_type=self.config.unit_type,
                        vocab=self.config.vocab_filepath)
                    self.config.output_dim = len(self.config.vocab_filepath)
                elif lang == 'zh':
                    self.text_feature = AutoTokenizer.from_pretrained(
                        self.config.tokenizer)
                    self.config.output_dim = self.text_feature.vocab_size
                self.config.decode.decoding_method = decode_method
            model_name = model_prefix[:model_prefix.rindex(
                '_')]  # model_type: {model_name}_{dataset}
        else:
            model_name = model_type
        model_class = self.task_resource.get_model_class(model_name)
        model_conf = self.config
        model = model_class.from_config(model_conf)
        self.model = model
        self.model.eval()

        # load model
        model_dict = paddle.load(self.ckpt_path)
        if task == 'asr':
            self.model.set_state_dict(model_dict)
        else:
            getattr(self.model, model_type).set_state_dict(model_dict)

    def preprocess(self, input: Union[str, os.PathLike]):
        """
        Input preprocess and return paddle.Tensor stored in self.input.
        Input content can be a text(tts), a file(asr, cls) or a streaming(not supported yet).
        """

        audio_file = input
        if isinstance(audio_file, (str, os.PathLike)):
            logger.debug("Preprocess audio_file:" + audio_file)
        elif isinstance(audio_file, io.BytesIO):
            audio_file.seek(0)

        # Get the object for feature extraction
        logger.debug("get the preprocess conf")
        preprocess_conf = self.config.preprocess_config
        preprocess_args = {"train": False}
        preprocessing = Transformation(preprocess_conf)
        logger.debug("read the audio file")
        audio, audio_sample_rate = soundfile.read(
            audio_file, dtype="int16", always_2d=True)
        if self.change_format:
            if audio.shape[1] >= 2:
                audio = audio.mean(axis=1, dtype=np.int16)
            else:
                audio = audio[:, 0]
            # pcm16 -> pcm 32
            audio = self._pcm16to32(audio)
            audio = librosa.resample(
                audio, orig_sr=audio_sample_rate, target_sr=self.sample_rate)
            audio_sample_rate = self.sample_rate
            # pcm32 -> pcm 16
            audio = self._pcm32to16(audio)
        else:
            audio = audio[:, 0]

        logger.debug(f"audio shape: {audio.shape}")
        # fbank
        audio = preprocessing(audio, **preprocess_args)

        audio_len = paddle.to_tensor(audio.shape[0]).unsqueeze(axis=0)
        audio = paddle.to_tensor(audio, dtype='float32').unsqueeze(axis=0)

        self._inputs["audio"] = audio
        self._inputs["audio_len"] = audio_len
        logger.debug(f"audio feat shape: {audio.shape}")

        logger.debug("audio feat process success")

    @paddle.no_grad()
    def infer(self, model_type: str, task: str):
        """
        Model inference and result stored in self.output.
        """
        logger.debug("start to infer the model to get the output")
        audio = self._inputs["audio"]
        if task == 'asr':
            cfg = self.config.decode
            logger.debug(f"we will use the {model_type}ASR like model.")
            try:
                result_transcripts = self.model.decode(
                    audio,
                    text_feature=self.text_feature,
                    decoding_method=cfg.decoding_method,
                    beam_size=cfg.beam_size,
                    tokenizer=getattr(self.config, 'tokenizer', None))
                self._outputs["result"] = result_transcripts[0][0]
            except Exception as e:
                logger.exception(e)
        else:
            logger.debug(
                f"we will use the {model_type} like model to extract audio feature."
            )
            try:
                out_feature = self.model(audio[:, :, 0])
                self._outputs["result"] = out_feature[0]
            except Exception as e:
                logger.exception(e)

    def postprocess(self) -> Union[str, os.PathLike]:
        """
            Output postprocess and return human-readable results such as texts and audio files.
        """
        return self._outputs["result"]

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

    def _check(self, audio_file: str, sample_rate: int, force_yes: bool=False):
        self.sample_rate = sample_rate
        if self.sample_rate != 16000 and self.sample_rate != 8000:
            logger.error(
                "invalid sample rate, please input --sr 8000 or --sr 16000")
            return False

        if isinstance(audio_file, (str, os.PathLike)):
            if not os.path.isfile(audio_file):
                logger.error("Please input the right audio file path")
                return False
        elif isinstance(audio_file, io.BytesIO):
            audio_file.seek(0)

        logger.debug("checking the audio file format......")
        try:
            audio, audio_sample_rate = soundfile.read(
                audio_file, dtype="int16", always_2d=True)
            audio_duration = audio.shape[0] / audio_sample_rate
            if audio_duration > self.max_len:
                logger.error(
                    f"Please input audio file less then {self.max_len} seconds.\n"
                )
                return False
        except Exception as e:
            logger.exception(e)
            logger.error(
                f"can not open the audio file, please check the audio file({audio_file}) format is 'wav'. \n \
                 you can try to use sox to change the file format.\n \
                 For example: \n \
                 sample rate: 16k \n \
                 sox input_audio.xx --rate 16k --bits 16 --channels 1 output_audio.wav \n \
                 sample rate: 8k \n \
                 sox input_audio.xx --rate 8k --bits 16 --channels 1 output_audio.wav \n \
                 ")
            return False
        logger.debug("The sample rate is %d" % audio_sample_rate)
        if audio_sample_rate != self.sample_rate:
            logger.warning("The sample rate of the input file is not {}.\n \
                            The program will resample the wav file to {}.\n \
                            If the result does not meet your expectations，\n \
                            Please input the 16k 16 bit 1 channel wav file. \
                        ".format(self.sample_rate, self.sample_rate))
            if force_yes is False:
                while (True):
                    logger.debug(
                        "Whether to change the sample rate and the channel. Y: change the sample. N: exit the prgream."
                    )
                    content = input("Input(Y/N):")
                    if content.strip() == "Y" or content.strip(
                    ) == "y" or content.strip() == "yes" or content.strip(
                    ) == "Yes":
                        logger.debug(
                            "change the sampele rate, channel to 16k and 1 channel"
                        )
                        break
                    elif content.strip() == "N" or content.strip(
                    ) == "n" or content.strip() == "no" or content.strip(
                    ) == "No":
                        logger.debug("Exit the program")
                        return False
                    else:
                        logger.warning("Not regular input, please input again")

            self.change_format = True
        else:
            logger.debug("The audio file format is right")
            self.change_format = False

        return True

    def execute(self, argv: List[str]) -> bool:
        """
            Command line entry.
        """
        parser_args = self.parser.parse_args(argv)

        model = parser_args.model
        task = parser_args.task
        lang = parser_args.lang
        sample_rate = parser_args.sample_rate
        config = parser_args.config
        ckpt_path = parser_args.ckpt_path
        decode_method = parser_args.decode_method
        force_yes = parser_args.yes
        rtf = parser_args.rtf
        device = parser_args.device

        if not parser_args.verbose:
            self.disable_task_loggers()

        task_source = self.get_input_source(parser_args.input)
        task_results = OrderedDict()
        has_exceptions = False

        for id_, input_ in task_source.items():
            try:
                res = self(
                    audio_file=input_,
                    model=model,
                    task=task,
                    lang=lang,
                    sample_rate=sample_rate,
                    config=config,
                    ckpt_path=ckpt_path,
                    decode_method=decode_method,
                    force_yes=force_yes,
                    rtf=rtf,
                    device=device)
                task_results[id_] = res

            except Exception as e:
                has_exceptions = True
                task_results[id_] = f'{e.__class__.__name__}: {e}'

        if rtf:
            self.show_rtf(CLI_TIMER[self.__class__.__name__])
        self.process_task_results(parser_args.input, task_results,
                                  parser_args.job_dump_result)
        if has_exceptions:
            return False
        else:
            return True

    @stats_wrapper
    def __call__(self,
                 audio_file: os.PathLike,
                 model: str=None,
                 task: str='asr',
                 lang: str='en',
                 sample_rate: int=16000,
                 config: os.PathLike=None,
                 ckpt_path: os.PathLike=None,
                 decode_method: str='ctc_greedy_search',
                 force_yes: bool=False,
                 rtf: bool=False,
                 device=paddle.get_device()):
        """
        Python API to call an executor.
        """

        current_call_params = {
            "model": model,
            "task": task,
            "lang": lang,
            "sample_rate": sample_rate,
            "config": config,
            "ckpt_path": ckpt_path,
            "decode_method": decode_method,
            "force_yes": force_yes,
            "rtf": rtf,
            "device": device
        }
        if self.last_call_params is not None and self.last_call_params != current_call_params and hasattr(
                self, 'model'):
            del self.model
        self.last_call_params = current_call_params

        audio_file = os.path.abspath(audio_file)
        paddle.set_device(device)
        self._init_from_path(model, task, lang, sample_rate, config,
                             decode_method, ckpt_path)
        if not self._check(audio_file, sample_rate, force_yes):
            sys.exit(-1)
        if rtf:
            k = self.__class__.__name__
            CLI_TIMER[k]['start'].append(time.time())
        self.preprocess(audio_file)
        self.infer(model, task)
        res = self.postprocess()  # Retrieve result of asr.

        if rtf:
            CLI_TIMER[k]['end'].append(time.time())
            audio, audio_sample_rate = soundfile.read(
                audio_file, dtype="int16", always_2d=True)
            CLI_TIMER[k]['extra'].append(audio.shape[0] / audio_sample_rate)

        return res
