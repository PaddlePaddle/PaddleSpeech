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
"""Evaluation for wav2vec2.0 model."""
import os
import sys
from pathlib import Path

import paddle
import soundfile
from paddlenlp.transformers import AutoTokenizer
from yacs.config import CfgNode

from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.models.wav2vec2.wav2vec2_ASR import Wav2vec2ASR
from paddlespeech.s2t.training.cli import default_argument_parser
from paddlespeech.s2t.utils.log import Log
from paddlespeech.s2t.utils.utility import UpdateConfig
logger = Log(__name__).getlog()


class Wav2vec2Infer():
    def __init__(self, config, args):
        self.args = args
        self.config = config
        self.audio_file = args.audio_file
        self.tokenizer = config.get("tokenizer", None)

        if self.tokenizer:
            self.text_feature = AutoTokenizer.from_pretrained(
                self.config.tokenizer)
        else:
            self.text_feature = TextFeaturizer(
                unit_type=config.unit_type, vocab=config.vocab_filepath)

        paddle.set_device('gpu' if self.args.ngpu > 0 else 'cpu')

        # model
        model_conf = config
        with UpdateConfig(model_conf):
            model_conf.output_dim = self.text_feature.vocab_size
        model = Wav2vec2ASR.from_config(model_conf)
        self.model = model
        self.model.eval()

        # load model
        params_path = self.args.checkpoint_path + ".pdparams"
        model_dict = paddle.load(params_path)
        self.model.set_state_dict(model_dict)

    def run(self):
        check(args.audio_file)

        with paddle.no_grad():
            # read
            audio, _ = soundfile.read(
                self.audio_file, dtype="int16", always_2d=True)
            logger.info(f"audio shape: {audio.shape}")
            xs = paddle.to_tensor(audio, dtype='float32').unsqueeze(axis=0)
            decode_config = self.config.decode
            result_transcripts, result_tokenids = self.model.decode(
                xs,
                text_feature=self.text_feature,
                decoding_method=decode_config.decoding_method,
                beam_size=decode_config.beam_size,
                tokenizer=self.tokenizer, )
            rsl = result_transcripts[0]
            utt = Path(self.audio_file).name
            logger.info(f"hyp: {utt} {rsl}")
            return rsl


def check(audio_file):
    if not os.path.isfile(audio_file):
        print("Please input the right audio file path")
        sys.exit(-1)

    logger.info("checking the audio file format......")
    try:
        sig, sample_rate = soundfile.read(audio_file)
    except Exception as e:
        logger.error(str(e))
        logger.error(
            "can not open the wav file, please check the audio file format")
        sys.exit(-1)
    logger.info("The sample rate is %d" % sample_rate)
    assert (sample_rate == 16000)
    logger.info("The audio file format is right")


def main(config, args):
    Wav2vec2Infer(config, args).run()


if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()

    config = CfgNode(new_allowed=True)

    if args.config:
        config.merge_from_file(args.config)
    if args.decode_cfg:
        decode_confs = CfgNode(new_allowed=True)
        decode_confs.merge_from_file(args.decode_cfg)
        config.decode = decode_confs
    if args.opts:
        config.merge_from_list(args.opts)
    config.freeze()
    main(config, args)
