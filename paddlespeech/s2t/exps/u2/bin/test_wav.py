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
"""Evaluation for U2 model."""
import os
import sys
from pathlib import Path

import numpy as np
import paddle
import soundfile

from paddlespeech.audio.transform.transformation import Transformation
from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.models.u2 import U2Model
from paddlespeech.s2t.training.cli import config_from_args
from paddlespeech.s2t.training.cli import default_argument_parser
from paddlespeech.s2t.utils.log import Log
from paddlespeech.s2t.utils.utility import UpdateConfig
logger = Log(__name__).getlog()

# TODO(hui zhang): dynamic load


class U2Infer():
    def __init__(self, config, args):
        self.args = args
        self.config = config
        self.audio_file = args.audio_file

        self.preprocess_conf = config.preprocess_config
        self.preprocess_args = {"train": False}
        self.preprocessing = Transformation(self.preprocess_conf)
        self.text_feature = TextFeaturizer(
            unit_type=config.unit_type,
            vocab=config.vocab_filepath,
            spm_model_prefix=config.spm_model_prefix)

        paddle.set_device('gpu' if self.args.ngpu > 0 else 'cpu')

        # model
        model_conf = config
        with UpdateConfig(model_conf):
            model_conf.input_dim = config.feat_dim
            model_conf.output_dim = self.text_feature.vocab_size
        model = U2Model.from_config(model_conf)
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
            audio, sample_rate = soundfile.read(
                self.audio_file, dtype="int16", always_2d=True)
            audio = audio[:, 0]
            logger.info(f"audio shape: {audio.shape}")

            # fbank
            feat = self.preprocessing(audio, **self.preprocess_args)
            logger.info(f"feat shape: {feat.shape}")
            if self.args.debug:
                np.savetxt("feat.transform.txt", feat)

            ilen = paddle.to_tensor(feat.shape[0]).unsqueeze(0)
            xs = paddle.to_tensor(feat, dtype='float32').unsqueeze(0)
            decode_config = self.config.decode
            logger.info(f"decode cfg: {decode_config}")
            reverse_weight = getattr(decode_config, 'reverse_weight', 0.0)
            result_transcripts = self.model.decode(
                xs,
                ilen,
                text_feature=self.text_feature,
                decoding_method=decode_config.decoding_method,
                beam_size=decode_config.beam_size,
                ctc_weight=decode_config.ctc_weight,
                decoding_chunk_size=decode_config.decoding_chunk_size,
                num_decoding_left_chunks=decode_config.num_decoding_left_chunks,
                simulate_streaming=decode_config.simulate_streaming,
                reverse_weight=reverse_weight)
            rsl = result_transcripts[0][0]
            utt = Path(self.audio_file).name
            logger.info(f"hyp: {utt} {result_transcripts[0][0]}")
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
    U2Infer(config, args).run()


if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()

    config = config_from_args(args)
    main(config, args)
