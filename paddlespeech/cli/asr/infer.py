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

import paddle
import soundfile

from ..executor import BaseExecutor
from ..utils import cli_register
from ..utils import download_and_decompress
from ..utils import logger
from ..utils import MODEL_HOME
from paddlespeech.s2t.exps.u2.config import get_cfg_defaults
from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.io.collator import SpeechCollator
from paddlespeech.s2t.transform.transformation import Transformation
from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.s2t.utils.utility import UpdateConfig

__all__ = ['ASRExecutor']

pretrained_models = {
    "wenetspeech_zh": {
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
            '--lang', type=str, default='zh', help='Choose model language.')
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
            default='cpu',
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
                        cfg_path: Optional[os.PathLike]=None,
                        ckpt_path: Optional[os.PathLike]=None):
        """
            Init model and other resources from a specific path.
        """
        if cfg_path is None or ckpt_path is None:
            tag = model_type + '_' + lang
            res_path = self._get_pretrained_path(tag)  # wenetspeech_zh
            self.cfg_path = os.path.join(res_path,
                                         pretrained_models[tag]['cfg_path'])
            self.ckpt_path = os.path.join(res_path,
                                          pretrained_models[tag]['ckpt_path'])
            logger.info(res_path)
            logger.info(self.cfg_path)
            logger.info(self.ckpt_path)
        else:
            self.cfg_path = os.path.abspath(cfg_path)
            self.ckpt_path = os.path.abspath(ckpt_path)
            res_path = os.path.dirname(
                os.path.dirname(os.path.abspath(self.cfg_path)))

        # Enter the path of model root
        os.chdir(res_path)

        #Init body.
        parser_args = self.parser_args
        paddle.set_device(parser_args.device)
        self.config = get_cfg_defaults()
        self.config.merge_from_file(self.cfg_path)
        self.config.decoding.decoding_method = "attention_rescoring"
        #self.config.freeze()
        model_conf = self.config.model
        logger.info(model_conf)

        with UpdateConfig(model_conf):
            if parser_args.model == "ds2_online" or parser_args.model == "ds2_offline":
                self.config.collator.vocab_filepath = os.path.join(
                    res_path, self.config.collator.vocab_filepath)
                self.config.collator.vocab_filepath = os.path.join(
                    res_path, self.config.collator.cmvn_path)
                self.collate_fn_test = SpeechCollator.from_config(self.config)
                model_conf.feat_size = self.collate_fn_test.feature_size
                model_conf.dict_size = self.text_feature.vocab_size
            elif parser_args.model == "conformer" or parser_args.model == "transformer" or parser_args.model == "wenetspeech":
                self.config.collator.vocab_filepath = os.path.join(
                    res_path, self.config.collator.vocab_filepath)
                self.text_feature = TextFeaturizer(
                    unit_type=self.config.collator.unit_type,
                    vocab_filepath=self.config.collator.vocab_filepath,
                    spm_model_prefix=self.config.collator.spm_model_prefix)
                model_conf.input_dim = self.config.collator.feat_dim
                model_conf.output_dim = self.text_feature.vocab_size
            else:
                raise Exception("wrong type")
        model_class = dynamic_import(parser_args.model, model_alias)
        model = model_class.from_config(model_conf)
        self.model = model
        self.model.eval()

        # load model
        params_path = self.ckpt_path + ".pdparams"
        model_dict = paddle.load(params_path)
        self.model.set_state_dict(model_dict)

    def preprocess(self, input: Union[str, os.PathLike]):
        """
            Input preprocess and return paddle.Tensor stored in self.input.
            Input content can be a text(tts), a file(asr, cls) or a streaming(not supported yet).
        """

        parser_args = self.parser_args
        config = self.config
        audio_file = input
        logger.info("audio_file" + audio_file)

        self.sr = config.collator.target_sample_rate

        # Get the object for feature extraction
        if parser_args.model == "ds2_online" or parser_args.model == "ds2_offline":
            audio, _ = collate_fn_test.process_utterance(
                audio_file=audio_file, transcript=" ")
            audio_len = audio.shape[0]
            audio = paddle.to_tensor(audio, dtype='float32')
            self.audio_len = paddle.to_tensor(audio_len)
            self.audio = paddle.unsqueeze(audio, axis=0)
            self.vocab_list = collate_fn_test.vocab_list
            logger.info(f"audio feat shape: {self.audio.shape}")

        elif parser_args.model == "conformer" or parser_args.model == "transformer" or parser_args.model == "wenetspeech":
            logger.info("get the preprocess conf")
            preprocess_conf = os.path.join(
                os.path.dirname(os.path.abspath(self.cfg_path)),
                "preprocess.yaml")

            cmvn_path: data / mean_std.json

            logger.info(preprocess_conf)
            preprocess_args = {"train": False}
            preprocessing = Transformation(preprocess_conf)
            audio, sample_rate = soundfile.read(
                audio_file, dtype="int16", always_2d=True)
            if sample_rate != self.sr:
                logger.error(
                    f"sample rate error: {sample_rate}, need {self.sr} ")
                sys.exit(-1)
            audio = audio[:, 0]
            logger.info(f"audio shape: {audio.shape}")
            # fbank
            audio = preprocessing(audio, **preprocess_args)

            self.audio_len = paddle.to_tensor(audio.shape[0])
            self.audio = paddle.to_tensor(
                audio, dtype='float32').unsqueeze(axis=0)
            logger.info(f"audio feat shape: {self.audio.shape}")

        else:
            raise Exception("wrong type")

    @paddle.no_grad()
    def infer(self):
        """
            Model inference and result stored in self.output.
        """
        cfg = self.config.decoding
        parser_args = self.parser_args
        audio = self.audio
        audio_len = self.audio_len
        if parser_args.model == "ds2_online" or parser_args.model == "ds2_offline":
            vocab_list = self.vocab_list
            result_transcripts = self.model.decode(
                audio,
                audio_len,
                vocab_list,
                decoding_method=cfg.decoding_method,
                lang_model_path=cfg.lang_model_path,
                beam_alpha=cfg.alpha,
                beam_beta=cfg.beta,
                beam_size=cfg.beam_size,
                cutoff_prob=cfg.cutoff_prob,
                cutoff_top_n=cfg.cutoff_top_n,
                num_processes=cfg.num_proc_bsearch)
            self.result_transcripts = result_transcripts[0]

        elif parser_args.model == "conformer" or parser_args.model == "transformer" or parser_args.model == "wenetspeech":
            text_feature = self.text_feature
            result_transcripts = self.model.decode(
                audio,
                audio_len,
                text_feature=self.text_feature,
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
            self.result_transcripts = result_transcripts[0][0]
        else:
            raise Exception("invalid model name")

        pass

    def postprocess(self) -> Union[str, os.PathLike]:
        """
            Output postprocess and return human-readable results such as texts and audio files.
        """
        return self.result_transcripts

    def execute(self, argv: List[str]) -> bool:
        """
            Command line entry.
        """
        self.parser_args = self.parser.parse_args(argv)

        model = self.parser_args.model
        lang = self.parser_args.lang
        config = self.parser_args.config
        ckpt_path = self.parser_args.ckpt_path
        audio_file = os.path.abspath(self.parser_args.input)
        device = self.parser_args.device

        try:
            res = self(model, lang, config, ckpt_path, audio_file, device)
            logger.info('ASR Result: {}'.format(res))
            return True
        except Exception as e:
            print(e)
            return False

    def __call__(self, model, lang, config, ckpt_path, audio_file, device):
        """
            Python API to call an executor.
        """
        self._init_from_path(model, lang, config, ckpt_path)
        self.preprocess(audio_file)
        self.infer()
        res = self.postprocess()  # Retrieve result of asr.

        return res
