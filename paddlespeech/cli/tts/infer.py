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
import time
from collections import OrderedDict
from typing import Any
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import paddle
import soundfile as sf
import yaml
from yacs.config import CfgNode

from ..executor import BaseExecutor
from ..log import logger
from ..utils import stats_wrapper
from paddlespeech.t2s.frontend import English
from paddlespeech.t2s.frontend.zh_frontend import Frontend
from paddlespeech.t2s.modules.normalizer import ZScore

__all__ = ['TTSExecutor']


class TTSExecutor(BaseExecutor):
    def __init__(self):
        super().__init__('tts')
        self.parser = argparse.ArgumentParser(
            prog='paddlespeech.tts', add_help=True)
        self.parser.add_argument(
            '--input', type=str, default=None, help='Input text to generate.')
        # acoustic model
        self.parser.add_argument(
            '--am',
            type=str,
            default='fastspeech2_csmsc',
            choices=[
                'speedyspeech_csmsc',
                'fastspeech2_csmsc',
                'fastspeech2_ljspeech',
                'fastspeech2_aishell3',
                'fastspeech2_vctk',
                'tacotron2_csmsc',
                'tacotron2_ljspeech',
            ],
            help='Choose acoustic model type of tts task.')
        self.parser.add_argument(
            '--am_config',
            type=str,
            default=None,
            help='Config of acoustic model. Use deault config when it is None.')
        self.parser.add_argument(
            '--am_ckpt',
            type=str,
            default=None,
            help='Checkpoint file of acoustic model.')
        self.parser.add_argument(
            "--am_stat",
            type=str,
            default=None,
            help="mean and standard deviation used to normalize spectrogram when training acoustic model."
        )
        self.parser.add_argument(
            "--phones_dict",
            type=str,
            default=None,
            help="phone vocabulary file.")
        self.parser.add_argument(
            "--tones_dict",
            type=str,
            default=None,
            help="tone vocabulary file.")
        self.parser.add_argument(
            "--speaker_dict",
            type=str,
            default=None,
            help="speaker id map file.")
        self.parser.add_argument(
            '--spk_id',
            type=int,
            default=0,
            help='spk id for multi speaker acoustic model')
        # vocoder
        self.parser.add_argument(
            '--voc',
            type=str,
            default='pwgan_csmsc',
            choices=[
                'pwgan_csmsc',
                'pwgan_ljspeech',
                'pwgan_aishell3',
                'pwgan_vctk',
                'mb_melgan_csmsc',
                'style_melgan_csmsc',
                'hifigan_csmsc',
                'hifigan_ljspeech',
                'hifigan_aishell3',
                'hifigan_vctk',
                'wavernn_csmsc',
            ],
            help='Choose vocoder type of tts task.')

        self.parser.add_argument(
            '--voc_config',
            type=str,
            default=None,
            help='Config of voc. Use deault config when it is None.')
        self.parser.add_argument(
            '--voc_ckpt',
            type=str,
            default=None,
            help='Checkpoint file of voc.')
        self.parser.add_argument(
            "--voc_stat",
            type=str,
            default=None,
            help="mean and standard deviation used to normalize spectrogram when training voc."
        )
        # other
        self.parser.add_argument(
            '--lang',
            type=str,
            default='zh',
            help='Choose model language. zh or en')
        self.parser.add_argument(
            '--device',
            type=str,
            default=paddle.get_device(),
            help='Choose device to execute model inference.')

        self.parser.add_argument(
            '--output', type=str, default='output.wav', help='output file name')
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

    def _init_from_path(
            self,
            am: str='fastspeech2_csmsc',
            am_config: Optional[os.PathLike]=None,
            am_ckpt: Optional[os.PathLike]=None,
            am_stat: Optional[os.PathLike]=None,
            phones_dict: Optional[os.PathLike]=None,
            tones_dict: Optional[os.PathLike]=None,
            speaker_dict: Optional[os.PathLike]=None,
            voc: str='pwgan_csmsc',
            voc_config: Optional[os.PathLike]=None,
            voc_ckpt: Optional[os.PathLike]=None,
            voc_stat: Optional[os.PathLike]=None,
            lang: str='zh', ):
        """
        Init model and other resources from a specific path.
        """
        if hasattr(self, 'am_inference') and hasattr(self, 'voc_inference'):
            logger.debug('Models had been initialized.')
            return

        # am
        if am_ckpt is None or am_config is None or am_stat is None or phones_dict is None:
            use_pretrained_am = True
        else:
            use_pretrained_am = False

        am_tag = am + '-' + lang
        self.task_resource.set_task_model(
            model_tag=am_tag,
            model_type=0,  # am
            skip_download=not use_pretrained_am,
            version=None,  # default version
        )
        if use_pretrained_am:
            self.am_res_path = self.task_resource.res_dir
            self.am_config = os.path.join(self.am_res_path,
                                          self.task_resource.res_dict['config'])
            self.am_ckpt = os.path.join(self.am_res_path,
                                        self.task_resource.res_dict['ckpt'])
            self.am_stat = os.path.join(
                self.am_res_path, self.task_resource.res_dict['speech_stats'])
            # must have phones_dict in acoustic
            self.phones_dict = os.path.join(
                self.am_res_path, self.task_resource.res_dict['phones_dict'])
            logger.debug(self.am_res_path)
            logger.debug(self.am_config)
            logger.debug(self.am_ckpt)
        else:
            self.am_config = os.path.abspath(am_config)
            self.am_ckpt = os.path.abspath(am_ckpt)
            self.am_stat = os.path.abspath(am_stat)
            self.phones_dict = os.path.abspath(phones_dict)
            self.am_res_path = os.path.dirname(os.path.abspath(self.am_config))

        # for speedyspeech
        self.tones_dict = None
        if 'tones_dict' in self.task_resource.res_dict:
            self.tones_dict = os.path.join(
                self.am_res_path, self.task_resource.res_dict['tones_dict'])
            if tones_dict:
                self.tones_dict = tones_dict

        # for multi speaker fastspeech2
        self.speaker_dict = None
        if 'speaker_dict' in self.task_resource.res_dict:
            self.speaker_dict = os.path.join(
                self.am_res_path, self.task_resource.res_dict['speaker_dict'])
            if speaker_dict:
                self.speaker_dict = speaker_dict

        # voc
        if voc_ckpt is None or voc_config is None or voc_stat is None:
            use_pretrained_voc = True
        else:
            use_pretrained_voc = False

        voc_tag = voc + '-' + lang
        self.task_resource.set_task_model(
            model_tag=voc_tag,
            model_type=1,  # vocoder
            skip_download=not use_pretrained_voc,
            version=None,  # default version
        )
        if use_pretrained_voc:
            self.voc_res_path = self.task_resource.voc_res_dir
            self.voc_config = os.path.join(
                self.voc_res_path, self.task_resource.voc_res_dict['config'])
            self.voc_ckpt = os.path.join(
                self.voc_res_path, self.task_resource.voc_res_dict['ckpt'])
            self.voc_stat = os.path.join(
                self.voc_res_path,
                self.task_resource.voc_res_dict['speech_stats'])
            logger.debug(self.voc_res_path)
            logger.debug(self.voc_config)
            logger.debug(self.voc_ckpt)
        else:
            self.voc_config = os.path.abspath(voc_config)
            self.voc_ckpt = os.path.abspath(voc_ckpt)
            self.voc_stat = os.path.abspath(voc_stat)
            self.voc_res_path = os.path.dirname(
                os.path.abspath(self.voc_config))

        # Init body.
        with open(self.am_config) as f:
            self.am_config = CfgNode(yaml.safe_load(f))
        with open(self.voc_config) as f:
            self.voc_config = CfgNode(yaml.safe_load(f))

        with open(self.phones_dict, "r") as f:
            phn_id = [line.strip().split() for line in f.readlines()]
        vocab_size = len(phn_id)

        tone_size = None
        if self.tones_dict:
            with open(self.tones_dict, "r") as f:
                tone_id = [line.strip().split() for line in f.readlines()]
            tone_size = len(tone_id)

        spk_num = None
        if self.speaker_dict:
            with open(self.speaker_dict, 'rt') as f:
                spk_id = [line.strip().split() for line in f.readlines()]
            spk_num = len(spk_id)

        # frontend
        if lang == 'zh':
            self.frontend = Frontend(
                phone_vocab_path=self.phones_dict,
                tone_vocab_path=self.tones_dict)

        elif lang == 'en':
            self.frontend = English(phone_vocab_path=self.phones_dict)

        # acoustic model
        odim = self.am_config.n_mels
        # model: {model_name}_{dataset}
        am_name = am[:am.rindex('_')]

        am_class = self.task_resource.get_model_class(am_name)
        am_inference_class = self.task_resource.get_model_class(am_name +
                                                                '_inference')

        if am_name == 'fastspeech2':
            am = am_class(
                idim=vocab_size,
                odim=odim,
                spk_num=spk_num,
                **self.am_config["model"])
        elif am_name == 'speedyspeech':
            am = am_class(
                vocab_size=vocab_size,
                tone_size=tone_size,
                **self.am_config["model"])
        elif am_name == 'tacotron2':
            am = am_class(idim=vocab_size, odim=odim, **self.am_config["model"])

        am.set_state_dict(paddle.load(self.am_ckpt)["main_params"])
        am.eval()
        am_mu, am_std = np.load(self.am_stat)
        am_mu = paddle.to_tensor(am_mu)
        am_std = paddle.to_tensor(am_std)
        am_normalizer = ZScore(am_mu, am_std)
        self.am_inference = am_inference_class(am_normalizer, am)
        self.am_inference.eval()

        # vocoder
        # model: {model_name}_{dataset}
        voc_name = voc[:voc.rindex('_')]
        voc_class = self.task_resource.get_model_class(voc_name)
        voc_inference_class = self.task_resource.get_model_class(voc_name +
                                                                 '_inference')
        if voc_name != 'wavernn':
            voc = voc_class(**self.voc_config["generator_params"])
            voc.set_state_dict(paddle.load(self.voc_ckpt)["generator_params"])
            voc.remove_weight_norm()
            voc.eval()
        else:
            voc = voc_class(**self.voc_config["model"])
            voc.set_state_dict(paddle.load(self.voc_ckpt)["main_params"])
            voc.eval()
        voc_mu, voc_std = np.load(self.voc_stat)
        voc_mu = paddle.to_tensor(voc_mu)
        voc_std = paddle.to_tensor(voc_std)
        voc_normalizer = ZScore(voc_mu, voc_std)
        self.voc_inference = voc_inference_class(voc_normalizer, voc)
        self.voc_inference.eval()

    def preprocess(self, input: Any, *args, **kwargs):
        """
        Input preprocess and return paddle.Tensor stored in self._inputs.
        Input content can be a text(tts), a file(asr, cls), a stream(not supported yet) or anything needed.

        Args:
            input (Any): Input text/file/stream or other content.
        """
        pass

    @paddle.no_grad()
    def infer(self,
              text: str,
              lang: str='zh',
              am: str='fastspeech2_csmsc',
              spk_id: int=0):
        """
        Model inference and result stored in self.output.
        """
        am_name = am[:am.rindex('_')]
        am_dataset = am[am.rindex('_') + 1:]
        get_tone_ids = False
        merge_sentences = False
        frontend_st = time.time()
        if am_name == 'speedyspeech':
            get_tone_ids = True
        if lang == 'zh':
            input_ids = self.frontend.get_input_ids(
                text,
                merge_sentences=merge_sentences,
                get_tone_ids=get_tone_ids)
            phone_ids = input_ids["phone_ids"]
            if get_tone_ids:
                tone_ids = input_ids["tone_ids"]
        elif lang == 'en':
            input_ids = self.frontend.get_input_ids(
                text, merge_sentences=merge_sentences)
            phone_ids = input_ids["phone_ids"]
        else:
            logger.error("lang should in {'zh', 'en'}!")
        self.frontend_time = time.time() - frontend_st

        self.am_time = 0
        self.voc_time = 0
        flags = 0
        for i in range(len(phone_ids)):
            am_st = time.time()
            part_phone_ids = phone_ids[i]
            # am
            if am_name == 'speedyspeech':
                part_tone_ids = tone_ids[i]
                mel = self.am_inference(part_phone_ids, part_tone_ids)
            # fastspeech2
            else:
                # multi speaker
                if am_dataset in {"aishell3", "vctk"}:
                    mel = self.am_inference(
                        part_phone_ids, spk_id=paddle.to_tensor(spk_id))
                else:
                    mel = self.am_inference(part_phone_ids)
            self.am_time += (time.time() - am_st)
            # voc
            voc_st = time.time()
            wav = self.voc_inference(mel)
            if flags == 0:
                wav_all = wav
                flags = 1
            else:
                wav_all = paddle.concat([wav_all, wav])
            self.voc_time += (time.time() - voc_st)
        self._outputs['wav'] = wav_all

    def postprocess(self, output: str='output.wav') -> Union[str, os.PathLike]:
        """
        Output postprocess and return results.
        This method get model output from self._outputs and convert it into human-readable results.

        Returns:
            Union[str, os.PathLike]: Human-readable results such as texts and audio files.
        """
        output = os.path.abspath(os.path.expanduser(output))
        sf.write(
            output, self._outputs['wav'].numpy(), samplerate=self.am_config.fs)
        return output

    def execute(self, argv: List[str]) -> bool:
        """
        Command line entry.
        """

        args = self.parser.parse_args(argv)

        am = args.am
        am_config = args.am_config
        am_ckpt = args.am_ckpt
        am_stat = args.am_stat
        phones_dict = args.phones_dict
        tones_dict = args.tones_dict
        speaker_dict = args.speaker_dict
        voc = args.voc
        voc_config = args.voc_config
        voc_ckpt = args.voc_ckpt
        voc_stat = args.voc_stat
        lang = args.lang
        device = args.device
        spk_id = args.spk_id

        if not args.verbose:
            self.disable_task_loggers()

        task_source = self.get_input_source(args.input)
        task_results = OrderedDict()
        has_exceptions = False

        for id_, input_ in task_source.items():
            if len(task_source) > 1:
                assert isinstance(args.output,
                                  str) and args.output.endswith('.wav')
                output = args.output.replace('.wav', f'_{id_}.wav')
            else:
                output = args.output

            try:
                res = self(
                    text=input_,
                    # acoustic model related
                    am=am,
                    am_config=am_config,
                    am_ckpt=am_ckpt,
                    am_stat=am_stat,
                    phones_dict=phones_dict,
                    tones_dict=tones_dict,
                    speaker_dict=speaker_dict,
                    spk_id=spk_id,
                    # vocoder related
                    voc=voc,
                    voc_config=voc_config,
                    voc_ckpt=voc_ckpt,
                    voc_stat=voc_stat,
                    # other
                    lang=lang,
                    device=device,
                    output=output)
                task_results[id_] = res
            except Exception as e:
                has_exceptions = True
                task_results[id_] = f'{e.__class__.__name__}: {e}'

        self.process_task_results(args.input, task_results,
                                  args.job_dump_result)

        if has_exceptions:
            return False
        else:
            return True

    @stats_wrapper
    def __call__(self,
                 text: str,
                 am: str='fastspeech2_csmsc',
                 am_config: Optional[os.PathLike]=None,
                 am_ckpt: Optional[os.PathLike]=None,
                 am_stat: Optional[os.PathLike]=None,
                 spk_id: int=0,
                 phones_dict: Optional[os.PathLike]=None,
                 tones_dict: Optional[os.PathLike]=None,
                 speaker_dict: Optional[os.PathLike]=None,
                 voc: str='pwgan_csmsc',
                 voc_config: Optional[os.PathLike]=None,
                 voc_ckpt: Optional[os.PathLike]=None,
                 voc_stat: Optional[os.PathLike]=None,
                 lang: str='zh',
                 device: str=paddle.get_device(),
                 output: str='output.wav'):
        """
        Python API to call an executor.
        """
        paddle.set_device(device)
        self._init_from_path(
            am=am,
            am_config=am_config,
            am_ckpt=am_ckpt,
            am_stat=am_stat,
            phones_dict=phones_dict,
            tones_dict=tones_dict,
            speaker_dict=speaker_dict,
            voc=voc,
            voc_config=voc_config,
            voc_ckpt=voc_ckpt,
            voc_stat=voc_stat,
            lang=lang)

        self.infer(text=text, lang=lang, am=am, spk_id=spk_id)

        res = self.postprocess(output=output)

        return res
