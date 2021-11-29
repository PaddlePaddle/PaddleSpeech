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

import paddle
import soundfile

from paddlespeech.s2t.exps.u2.config import get_cfg_defaults
from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.models.u2 import U2Model
from paddlespeech.s2t.training.cli import default_argument_parser
from paddlespeech.s2t.transform.transformation import Transformation
from paddlespeech.s2t.utils.log import Log
from paddlespeech.s2t.utils.utility import UpdateConfig
logger = Log(__name__).getlog()

# TODO(hui zhang): dynamic load


class U2Infer():
    def __init__(self, config, args):
        self.args = args
        self.config = config
        self.audio_file = args.audio_file
        self.sr = config.collator.target_sample_rate

        self.preprocess_conf = config.collator.augmentation_config
        self.preprocess_args = {"train": False}
        self.preprocessing = Transformation(self.preprocess_conf)

        self.text_feature = TextFeaturizer(
            unit_type=config.collator.unit_type,
            vocab_filepath=config.collator.vocab_filepath,
            spm_model_prefix=config.collator.spm_model_prefix)

        paddle.set_device('gpu' if self.args.ngpu > 0 else 'cpu')

        # model
        model_conf = config.model
        with UpdateConfig(model_conf):
            model_conf.input_dim = config.collator.feat_dim
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
            if sample_rate != self.sr:
                logger.error(
                    f"sample rate error: {sample_rate}, need {self.sr} ")
                sys.exit(-1)

            audio = audio[:, 0]
            logger.info(f"audio shape: {audio.shape}")

            # fbank
            feat = self.preprocessing(audio, **self.preprocess_args)
            logger.info(f"feat shape: {feat.shape}")

            ilen = paddle.to_tensor(feat.shape[0])
            xs = paddle.to_tensor(feat, dtype='float32').unsqueeze(axis=0)

            cfg = self.config.decoding
            result_transcripts = self.model.decode(
                xs,
                ilen,
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
    # save asr result to
    parser.add_argument(
        "--result_file", type=str, help="path of save the asr result")
    parser.add_argument(
        "--audio_file", type=str, help="path of the input audio file")
    args = parser.parse_args()

    config = get_cfg_defaults()
    if args.config:
        config.merge_from_file(args.config)
    if args.opts:
        config.merge_from_list(args.opts)
    config.freeze()
    main(config, args)
