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
from ..utils import cli_register
from ..utils import download_and_decompress
from ..utils import logger
from ..utils import MODEL_HOME
from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.s2t.utils.utility import UpdateConfig

__all__ = ["STExecutor"]

pretrained_models = {
    "fat_st_ted-en-zh": {
        "url":
        "https://paddlespeech.bj.bcebos.com/s2t/ted_en_zh/st1/fat_st_ted-en-zh.tar.gz",
        "md5":
        "fa0a7425b91b4f8d259c70b2aca5ae67",
        "cfg_path":
        "conf/transformer_mtl_noam.yaml",
        "ckpt_path":
        "exp/transformer_mtl_noam/checkpoints/fat_st_ted-en-zh.pdparams",
    }
}

model_alias = {"fat_st": "paddlespeech.s2t.models.u2_st:U2STModel"}

kaldi_bins = {
    "url":
    "https://paddlespeech.bj.bcebos.com/s2t/ted_en_zh/st1/kaldi_bins.tar.gz",
    "md5":
    "c0682303b3f3393dbf6ed4c4e35a53eb",
}


@cli_register(
    name="paddlespeech.st", description="Speech translation infer command.")
class STExecutor(BaseExecutor):
    def __init__(self):
        super(STExecutor, self).__init__()

        self.parser = argparse.ArgumentParser(
            prog="paddlespeech.st", add_help=True)
        self.parser.add_argument(
            "--input", type=str, required=True, help="Audio file to translate.")
        self.parser.add_argument(
            "--model",
            type=str,
            default="fat_st_ted",
            choices=[tag[:tag.index('-')] for tag in pretrained_models.keys()],
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

    def _get_pretrained_path(self, tag: str) -> os.PathLike:
        """
            Download and returns pretrained resources path of current task.
        """
        assert tag in pretrained_models, "Can not find pretrained resources of {}.".format(
            tag)

        res_path = os.path.join(MODEL_HOME, tag)
        decompressed_path = download_and_decompress(pretrained_models[tag],
                                                    res_path)
        decompressed_path = os.path.abspath(decompressed_path)
        logger.info(
            "Use pretrained model stored in: {}".format(decompressed_path))

        return decompressed_path

    def _set_kaldi_bins(self) -> os.PathLike:
        """
            Download and returns kaldi_bins resources path of current task.
        """
        decompressed_path = download_and_decompress(kaldi_bins, MODEL_HOME)
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
            res_path = self._get_pretrained_path(tag)
            self.cfg_path = os.path.join(res_path,
                                         pretrained_models[tag]["cfg_path"])
            self.ckpt_path = os.path.join(res_path,
                                          pretrained_models[tag]["ckpt_path"])
            logger.info(res_path)
            logger.info(self.cfg_path)
            logger.info(self.ckpt_path)
        else:
            self.cfg_path = os.path.abspath(cfg_path)
            self.ckpt_path = os.path.abspath(ckpt_path)
            res_path = os.path.dirname(
                os.path.dirname(os.path.abspath(self.cfg_path)))

        #Init body.
        self.config = CfgNode(new_allowed=True)
        self.config.merge_from_file(self.cfg_path)
        self.config.decoding.decoding_method = "fullsentence"

        with UpdateConfig(self.config):
            self.config.collator.vocab_filepath = os.path.join(
                res_path, self.config.collator.vocab_filepath)
            self.config.collator.cmvn_path = os.path.join(
                res_path, self.config.collator.cmvn_path)
            self.config.collator.spm_model_prefix = os.path.join(
                res_path, self.config.collator.spm_model_prefix)
            self.text_feature = TextFeaturizer(
                unit_type=self.config.collator.unit_type,
                vocab_filepath=self.config.collator.vocab_filepath,
                spm_model_prefix=self.config.collator.spm_model_prefix)
            self.config.model.input_dim = self.config.collator.feat_dim
            self.config.model.output_dim = self.text_feature.vocab_size

        model_conf = self.config.model
        logger.info(model_conf)
        model_name = model_type[:model_type.rindex(
            '_')]  # model_type: {model_name}_{dataset}
        model_class = dynamic_import(model_name, model_alias)
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
            cmvn = self.config.collator.cmvn_path
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
        cfg = self.config.decoding
        audio = self._inputs["audio"]
        audio_len = self._inputs["audio_len"]
        if model_type == "fat_st_ted":
            hyps = self.model.decode(
                audio,
                audio_len,
                text_feature=self.text_feature,
                decoding_method=cfg.decoding_method,
                lang_model_path=None,
                beam_alpha=cfg.alpha,
                beam_beta=cfg.beta,
                beam_size=cfg.beam_size,
                cutoff_prob=cfg.cutoff_prob,
                cutoff_top_n=cfg.cutoff_top_n,
                num_processes=cfg.num_proc_bsearch,
                ctc_weight=cfg.ctc_weight,
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
        audio_file = parser_args.input
        device = parser_args.device

        try:
            res = self(model, src_lang, tgt_lang, sample_rate, config,
                       ckpt_path, audio_file, device)
            logger.info("ST Result: {}".format(res))
            return True
        except Exception as e:
            logger.exception(e)
            return False

    def __call__(self, model, src_lang, tgt_lang, sample_rate, config,
                 ckpt_path, audio_file, device):
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
