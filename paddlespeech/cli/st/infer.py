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
import subprocess
from collections import OrderedDict
from typing import List
from typing import Optional
from typing import Union

import kaldiio
import numpy as np
import paddle
import soundfile
from kaldiio import WriteHelper
from yacs.config import CfgNode

from ..executor import BaseExecutor
from ..log import logger
from ..utils import download_and_decompress
from ..utils import MODEL_HOME
from ..utils import stats_wrapper
from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.utils.utility import UpdateConfig

__all__ = ["STExecutor"]

kaldi_bins = {
    "url":
    "https://paddlespeech.bj.bcebos.com/s2t/ted_en_zh/st1/kaldi_bins.tar.gz",
    "md5":
    "c0682303b3f3393dbf6ed4c4e35a53eb",
}


class STExecutor(BaseExecutor):
    def __init__(self):
        super().__init__(task='st')
        self.kaldi_bins = kaldi_bins

        self.parser = argparse.ArgumentParser(
            prog="paddlespeech.st", add_help=True)
        self.parser.add_argument(
            "--input", type=str, default=None, help="Audio file to translate.")
        self.parser.add_argument(
            "--model",
            type=str,
            default="fat_st_ted",
            choices=[
                tag[:tag.index('-')]
                for tag in self.task_resource.pretrained_models.keys()
            ],
            help="Choose model type of st task.")
        self.parser.add_argument(
            "--src_lang",
            type=str,
            default="en",
            help="Choose model source language.")
        self.parser.add_argument(
            "--tgt_lang",
            type=str,
            default="zh",
            help="Choose model target language.")
        self.parser.add_argument(
            "--sample_rate",
            type=int,
            default=16000,
            choices=[16000],
            help='Choose the audio sample rate of the model. 8000 or 16000')
        self.parser.add_argument(
            "--config",
            type=str,
            default=None,
            help="Config of st task. Use deault config when it is None.")
        self.parser.add_argument(
            "--ckpt_path",
            type=str,
            default=None,
            help="Checkpoint file of model.")
        self.parser.add_argument(
            "--device",
            type=str,
            default=paddle.get_device(),
            help="Choose device to execute model inference.")
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

    def _set_kaldi_bins(self) -> os.PathLike:
        """
            Download and returns kaldi_bins resources path of current task.
        """
        decompressed_path = download_and_decompress(self.kaldi_bins, MODEL_HOME)
        decompressed_path = os.path.abspath(decompressed_path)
        logger.info("Kaldi_bins stored in: {}".format(decompressed_path))
        if "LD_LIBRARY_PATH" in os.environ:
            os.environ["LD_LIBRARY_PATH"] += f":{decompressed_path}"
        else:
            os.environ["LD_LIBRARY_PATH"] = f"{decompressed_path}"
        os.environ["PATH"] += f":{decompressed_path}"
        return decompressed_path

    def _init_from_path(self,
                        model_type: str="fat_st_ted",
                        src_lang: str="en",
                        tgt_lang: str="zh",
                        cfg_path: Optional[os.PathLike]=None,
                        ckpt_path: Optional[os.PathLike]=None):
        """
            Init model and other resources from a specific path.
        """
        if hasattr(self, 'model'):
            logger.info('Model had been initialized.')
            return

        if cfg_path is None or ckpt_path is None:
            tag = model_type + "-" + src_lang + "-" + tgt_lang
            self.task_resource.set_task_model(tag, version=None)
            self.cfg_path = os.path.join(
                self.task_resource.res_dir,
                self.task_resource.res_dict['cfg_path'])
            self.ckpt_path = os.path.join(
                self.task_resource.res_dir,
                self.task_resource.res_dict['ckpt_path'])
            logger.info(self.cfg_path)
            logger.info(self.ckpt_path)
            res_path = self.task_resource.res_dir
        else:
            self.cfg_path = os.path.abspath(cfg_path)
            self.ckpt_path = os.path.abspath(ckpt_path)
            res_path = os.path.dirname(
                os.path.dirname(os.path.abspath(self.cfg_path)))

        #Init body.
        self.config = CfgNode(new_allowed=True)
        self.config.merge_from_file(self.cfg_path)
        self.config.decode.decoding_method = "fullsentence"

        with UpdateConfig(self.config):
            self.config.cmvn_path = os.path.join(res_path,
                                                 self.config.cmvn_path)
            self.config.spm_model_prefix = os.path.join(
                res_path, self.config.spm_model_prefix)
            self.text_feature = TextFeaturizer(
                unit_type=self.config.unit_type,
                vocab=self.config.vocab_filepath,
                spm_model_prefix=self.config.spm_model_prefix)

        model_conf = self.config
        model_name = model_type[:model_type.rindex(
            '_')]  # model_type: {model_name}_{dataset}
        model_class = self.task_resource.get_model_class(model_name)
        self.model = model_class.from_config(model_conf)
        self.model.eval()

        # load model
        params_path = self.ckpt_path
        model_dict = paddle.load(params_path)
        self.model.set_state_dict(model_dict)

        # set kaldi bins
        self._set_kaldi_bins()

    def _check(self, audio_file: str, sample_rate: int):
        _, audio_sample_rate = soundfile.read(
            audio_file, dtype="int16", always_2d=True)
        if audio_sample_rate != sample_rate:
            raise Exception("invalid sample rate")
            sys.exit(-1)

    def preprocess(self, wav_file: Union[str, os.PathLike], model_type: str):
        """
            Input preprocess and return paddle.Tensor stored in self.input.
            Input content can be a file(wav).
        """
        audio_file = os.path.abspath(wav_file)
        logger.info("Preprocess audio_file:" + audio_file)

        if "fat_st" in model_type:
            cmvn = self.config.cmvn_path
            utt_name = "_tmp"

            # Get the object for feature extraction
            fbank_extract_command = [
                "compute-fbank-feats", "--num-mel-bins=80", "--verbose=2",
                "--sample-frequency=16000", "scp:-", "ark:-"
            ]
            fbank_extract_process = subprocess.Popen(
                fbank_extract_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            fbank_extract_process.stdin.write(
                f"{utt_name} {wav_file}".encode("utf8"))
            fbank_extract_process.stdin.close()
            fbank_feat = dict(
                kaldiio.load_ark(fbank_extract_process.stdout))[utt_name]

            extract_command = ["compute-kaldi-pitch-feats", "scp:-", "ark:-"]
            pitch_extract_process = subprocess.Popen(
                extract_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            pitch_extract_process.stdin.write(
                f"{utt_name} {wav_file}".encode("utf8"))
            process_command = ["process-kaldi-pitch-feats", "ark:", "ark:-"]
            pitch_process = subprocess.Popen(
                process_command,
                stdin=pitch_extract_process.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            pitch_extract_process.stdin.close()
            pitch_feat = dict(kaldiio.load_ark(pitch_process.stdout))[utt_name]
            concated_feat = np.concatenate((fbank_feat, pitch_feat), axis=1)
            raw_feat = f"{utt_name}.raw"
            with WriteHelper(
                    f"ark,scp:{raw_feat}.ark,{raw_feat}.scp") as writer:
                writer(utt_name, concated_feat)
            cmvn_command = [
                "apply-cmvn", "--norm-vars=true", cmvn, f"scp:{raw_feat}.scp",
                "ark:-"
            ]
            cmvn_process = subprocess.Popen(
                cmvn_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process_command = [
                "copy-feats", "--compress=true", "ark:-", "ark:-"
            ]
            process = subprocess.Popen(
                process_command,
                stdin=cmvn_process.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            norm_feat = dict(kaldiio.load_ark(process.stdout))[utt_name]
            self._inputs["audio"] = paddle.to_tensor(norm_feat).unsqueeze(0)
            self._inputs["audio_len"] = paddle.to_tensor(
                self._inputs["audio"].shape[1], dtype="int64")
        else:
            raise ValueError("Wrong model type.")

    @paddle.no_grad()
    def infer(self, model_type: str):
        """
            Model inference and result stored in self.output.
        """
        cfg = self.config.decode
        audio = self._inputs["audio"]
        audio_len = self._inputs["audio_len"]
        if model_type == "fat_st_ted":
            hyps = self.model.decode(
                audio,
                audio_len,
                text_feature=self.text_feature,
                decoding_method=cfg.decoding_method,
                beam_size=cfg.beam_size,
                word_reward=cfg.word_reward,
                decoding_chunk_size=cfg.decoding_chunk_size,
                num_decoding_left_chunks=cfg.num_decoding_left_chunks,
                simulate_streaming=cfg.simulate_streaming)
            self._outputs["result"] = hyps
        else:
            raise ValueError("Wrong model type.")

    def postprocess(self, model_type: str) -> Union[str, os.PathLike]:
        """
            Output postprocess and return human-readable results such as texts and audio files.
        """
        if model_type == "fat_st_ted":
            return self._outputs["result"]
        else:
            raise ValueError("Wrong model type.")

    def execute(self, argv: List[str]) -> bool:
        """
            Command line entry.
        """
        parser_args = self.parser.parse_args(argv)

        model = parser_args.model
        src_lang = parser_args.src_lang
        tgt_lang = parser_args.tgt_lang
        sample_rate = parser_args.sample_rate
        config = parser_args.config
        ckpt_path = parser_args.ckpt_path
        device = parser_args.device

        if not parser_args.verbose:
            self.disable_task_loggers()

        task_source = self.get_input_source(parser_args.input)
        task_results = OrderedDict()
        has_exceptions = False

        for id_, input_ in task_source.items():
            try:
                res = self(input_, model, src_lang, tgt_lang, sample_rate,
                           config, ckpt_path, device)
                task_results[id_] = res
            except Exception as e:
                has_exceptions = True
                task_results[id_] = f'{e.__class__.__name__}: {e}'

        self.process_task_results(parser_args.input, task_results,
                                  parser_args.job_dump_result)

        if has_exceptions:
            return False
        else:
            return True

    @stats_wrapper
    def __call__(self,
                 audio_file: os.PathLike,
                 model: str='fat_st_ted',
                 src_lang: str='en',
                 tgt_lang: str='zh',
                 sample_rate: int=16000,
                 config: Optional[os.PathLike]=None,
                 ckpt_path: Optional[os.PathLike]=None,
                 device: str=paddle.get_device()):
        """
            Python API to call an executor.
        """
        audio_file = os.path.abspath(audio_file)
        self._check(audio_file, sample_rate)
        paddle.set_device(device)
        self._init_from_path(model, src_lang, tgt_lang, config, ckpt_path)
        self.preprocess(audio_file, model)
        self.infer(model)
        res = self.postprocess(model)

        return res
