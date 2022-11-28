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
from paddlespeech.resource import CommonTaskResource
from paddlespeech.t2s.exps.syn_utils import get_am_inference
from paddlespeech.t2s.exps.syn_utils import get_frontend
from paddlespeech.t2s.exps.syn_utils import get_sess
from paddlespeech.t2s.exps.syn_utils import get_voc_inference
from paddlespeech.t2s.exps.syn_utils import run_frontend
from paddlespeech.t2s.utils import str2bool

__all__ = ['TTSExecutor']
ONNX_SUPPORT_SET = {
    'speedyspeech_csmsc', 'fastspeech2_csmsc', 'fastspeech2_ljspeech',
    'fastspeech2_aishell3', 'fastspeech2_vctk', 'pwgan_csmsc', 'pwgan_ljspeech',
    'pwgan_aishell3', 'pwgan_vctk', 'mb_melgan_csmsc', 'hifigan_csmsc',
    'hifigan_ljspeech', 'hifigan_aishell3', 'hifigan_vctk'
}


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
                'fastspeech2_mix',
                'tacotron2_csmsc',
                'tacotron2_ljspeech',
                'fastspeech2_male',
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
            default='hifigan_csmsc',
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
                'pwgan_male',
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
            help='Choose model language. zh or en or mix')
        self.parser.add_argument(
            '--device',
            type=str,
            default=paddle.get_device(),
            help='Choose device to execute model inference.')

        self.parser.add_argument('--cpu_threads', type=int, default=2)

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
        self.parser.add_argument(
            "--use_onnx",
            type=str2bool,
            default=False,
            help="whether to usen onnxruntime inference.")
        self.parser.add_argument(
            '--fs',
            type=int,
            default=24000,
            help='sample rate for onnx models when use specified model files.')

    def _init_from_path(
            self,
            am: str='fastspeech2_csmsc',
            am_config: Optional[os.PathLike]=None,
            am_ckpt: Optional[os.PathLike]=None,
            am_stat: Optional[os.PathLike]=None,
            phones_dict: Optional[os.PathLike]=None,
            tones_dict: Optional[os.PathLike]=None,
            speaker_dict: Optional[os.PathLike]=None,
            voc: str='hifigan_csmsc',
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
            self.am_res_path = os.path.dirname(self.am_config)

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
        voc_lang = lang
        # When speaker is 174 (csmsc), use csmsc's vocoder is better than aishell3's
        if lang == 'mix':
            voc_lang = 'zh'
        voc_tag = voc + '-' + voc_lang
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
        self.frontend = get_frontend(
            lang=lang, phones_dict=self.phones_dict, tones_dict=self.tones_dict)

        # acoustic model
        self.am_inference = get_am_inference(
            am=am,
            am_config=self.am_config,
            am_ckpt=self.am_ckpt,
            am_stat=self.am_stat,
            phones_dict=self.phones_dict,
            tones_dict=self.tones_dict,
            speaker_dict=self.speaker_dict)

        # vocoder
        self.voc_inference = get_voc_inference(
            voc=voc,
            voc_config=self.voc_config,
            voc_ckpt=self.voc_ckpt,
            voc_stat=self.voc_stat)

    def _init_from_path_onnx(self,
                             am: str='fastspeech2_csmsc',
                             am_ckpt: Optional[os.PathLike]=None,
                             phones_dict: Optional[os.PathLike]=None,
                             tones_dict: Optional[os.PathLike]=None,
                             speaker_dict: Optional[os.PathLike]=None,
                             voc: str='hifigan_csmsc',
                             voc_ckpt: Optional[os.PathLike]=None,
                             lang: str='zh',
                             device: str='cpu',
                             cpu_threads: int=2,
                             fs: int=24000):
        if hasattr(self, 'am_sess') and hasattr(self, 'voc_sess'):
            logger.debug('Models had been initialized.')
            return

        # am
        if am_ckpt is None or phones_dict is None:
            use_pretrained_am = True
        else:
            use_pretrained_am = False

        am_tag = am + '_onnx' + '-' + lang
        self.task_resource.set_task_model(
            model_tag=am_tag,
            model_type=0,  # am
            skip_download=not use_pretrained_am,
            version=None,  # default version
        )
        if use_pretrained_am:
            self.am_res_path = self.task_resource.res_dir
            self.am_ckpt = os.path.join(self.am_res_path,
                                        self.task_resource.res_dict['ckpt'])
            # must have phones_dict in acoustic
            self.phones_dict = os.path.join(
                self.am_res_path, self.task_resource.res_dict['phones_dict'])
            self.am_fs = self.task_resource.res_dict['sample_rate']
            logger.debug(self.am_res_path)
            logger.debug(self.am_ckpt)
        else:
            self.am_ckpt = os.path.abspath(am_ckpt)
            self.phones_dict = os.path.abspath(phones_dict)
            self.am_res_path = os.path.dirname(self.am_ckpt)
            self.am_fs = fs

        # for speedyspeech
        self.tones_dict = None
        if 'tones_dict' in self.task_resource.res_dict:
            self.tones_dict = os.path.join(
                self.am_res_path, self.task_resource.res_dict['tones_dict'])
            if tones_dict:
                self.tones_dict = tones_dict

        # voc
        if voc_ckpt is None:
            use_pretrained_voc = True
        else:
            use_pretrained_voc = False
        voc_lang = lang
        # we must use ljspeech's voc for mix am now!
        if lang == 'mix':
            voc_lang = 'en'
        voc_tag = voc + '_onnx' + '-' + voc_lang
        self.task_resource.set_task_model(
            model_tag=voc_tag,
            model_type=1,  # vocoder
            skip_download=not use_pretrained_voc,
            version=None,  # default version
        )
        if use_pretrained_voc:
            self.voc_res_path = self.task_resource.voc_res_dir
            self.voc_ckpt = os.path.join(
                self.voc_res_path, self.task_resource.voc_res_dict['ckpt'])
            logger.debug(self.voc_res_path)
            logger.debug(self.voc_ckpt)
        else:
            self.voc_ckpt = os.path.abspath(voc_ckpt)
            self.voc_res_path = os.path.dirname(os.path.abspath(self.voc_ckpt))

        # frontend
        self.frontend = get_frontend(
            lang=lang, phones_dict=self.phones_dict, tones_dict=self.tones_dict)
        self.am_sess = get_sess(
            model_path=self.am_ckpt, device=device, cpu_threads=cpu_threads)

        # vocoder
        self.voc_sess = get_sess(
            model_path=self.voc_ckpt, device=device, cpu_threads=cpu_threads)

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
        merge_sentences = False
        get_tone_ids = False
        if am_name == 'speedyspeech':
            get_tone_ids = True
        frontend_st = time.time()
        frontend_dict = run_frontend(
            frontend=self.frontend,
            text=text,
            merge_sentences=merge_sentences,
            get_tone_ids=get_tone_ids,
            lang=lang)
        self.frontend_time = time.time() - frontend_st
        self.am_time = 0
        self.voc_time = 0
        flags = 0
        phone_ids = frontend_dict['phone_ids']
        for i in range(len(phone_ids)):
            am_st = time.time()
            part_phone_ids = phone_ids[i]
            # am
            if am_name == 'speedyspeech':
                part_tone_ids = frontend_dict['tone_ids'][i]
                mel = self.am_inference(part_phone_ids, part_tone_ids)
            # fastspeech2
            else:
                # multi speaker
                if am_dataset in {'aishell3', 'vctk', 'mix'}:
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

    def infer_onnx(self,
                   text: str,
                   lang: str='zh',
                   am: str='fastspeech2_csmsc',
                   spk_id: int=0):
        am_name = am[:am.rindex('_')]
        am_dataset = am[am.rindex('_') + 1:]
        merge_sentences = False
        get_tone_ids = False
        if am_name == 'speedyspeech':
            get_tone_ids = True
        am_input_feed = {}
        frontend_st = time.time()
        frontend_dict = run_frontend(
            frontend=self.frontend,
            text=text,
            merge_sentences=merge_sentences,
            get_tone_ids=get_tone_ids,
            lang=lang,
            to_tensor=False)
        self.frontend_time = time.time() - frontend_st
        phone_ids = frontend_dict['phone_ids']
        self.am_time = 0
        self.voc_time = 0
        flags = 0
        for i in range(len(phone_ids)):
            am_st = time.time()
            part_phone_ids = phone_ids[i]
            if am_name == 'fastspeech2':
                am_input_feed.update({'text': part_phone_ids})
                if am_dataset in {"aishell3", "vctk"}:
                    # NOTE: 'spk_id' should be List[int] rather than int here!!
                    am_input_feed.update({'spk_id': [spk_id]})
            elif am_name == 'speedyspeech':
                part_tone_ids = frontend_dict['tone_ids'][i]
                am_input_feed.update({
                    'phones': part_phone_ids,
                    'tones': part_tone_ids
                })
            mel = self.am_sess.run(output_names=None, input_feed=am_input_feed)
            mel = mel[0]
            self.am_time += (time.time() - am_st)
            # voc
            voc_st = time.time()
            wav = self.voc_sess.run(
                output_names=None, input_feed={'logmel': mel})
            wav = wav[0]
            if flags == 0:
                wav_all = wav
                flags = 1
            else:
                wav_all = np.concatenate([wav_all, wav])
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

    def postprocess_onnx(self,
                         output: str='output.wav') -> Union[str, os.PathLike]:
        """
        Output postprocess and return results.
        This method get model output from self._outputs and convert it into human-readable results.

        Returns:
            Union[str, os.PathLike]: Human-readable results such as texts and audio files.
        """
        output = os.path.abspath(os.path.expanduser(output))
        sf.write(output, self._outputs['wav'], samplerate=self.am_fs)
        return output

    # 命令行的入口是这里
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
        use_onnx = args.use_onnx
        cpu_threads = args.cpu_threads
        fs = args.fs

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
                    output=output,
                    use_onnx=use_onnx,
                    cpu_threads=cpu_threads,
                    fs=fs)
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

    # pyton api 的入口是这里
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
                 voc: str='hifigan_csmsc',
                 voc_config: Optional[os.PathLike]=None,
                 voc_ckpt: Optional[os.PathLike]=None,
                 voc_stat: Optional[os.PathLike]=None,
                 lang: str='zh',
                 device: str=paddle.get_device(),
                 output: str='output.wav',
                 use_onnx: bool=False,
                 cpu_threads: int=2,
                 fs: int=24000):
        """
        Python API to call an executor.
        """
        if not use_onnx:
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
        else:
            # use onnx
            # we use `cpu` for onnxruntime by default
            # please see description in https://github.com/PaddlePaddle/PaddleSpeech/pull/2220
            self.task_resource = CommonTaskResource(
                task='tts', model_format='onnx')
            assert (
                am in ONNX_SUPPORT_SET and voc in ONNX_SUPPORT_SET
            ), f'the am and voc you choose, they should be in {ONNX_SUPPORT_SET}'
            self._init_from_path_onnx(
                am=am,
                am_ckpt=am_ckpt,
                phones_dict=phones_dict,
                tones_dict=tones_dict,
                speaker_dict=speaker_dict,
                voc=voc,
                voc_ckpt=voc_ckpt,
                lang=lang,
                device=device,
                cpu_threads=cpu_threads,
                fs=fs)
            self.infer_onnx(text=text, lang=lang, am=am, spk_id=spk_id)
            res = self.postprocess_onnx(output=output)
            return res
