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
import cProfile
import os
import sys

import paddle
import soundfile

from paddlespeech.s2t.exps.u2.config import get_cfg_defaults
from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.io.collator import SpeechCollator
from paddlespeech.s2t.models.u2 import U2Model
from paddlespeech.s2t.training.cli import default_argument_parser
from paddlespeech.s2t.training.trainer import Trainer
from paddlespeech.s2t.utils import layer_tools
from paddlespeech.s2t.utils import mp_tools
from paddlespeech.s2t.utils.log import Log
from paddlespeech.s2t.utils.utility import print_arguments
from paddlespeech.s2t.utils.utility import UpdateConfig
logger = Log(__name__).getlog()

# TODO(hui zhang): dynamic load


class U2Tester_Hub(Trainer):
    def __init__(self, config, args):
        # super().__init__(config, args)
        self.args = args
        self.config = config
        self.audio_file = args.audio_file
        self.collate_fn_test = SpeechCollator.from_config(config)
        self._text_featurizer = TextFeaturizer(
            unit_type=config.collator.unit_type,
            vocab_filepath=None,
            spm_model_prefix=config.collator.spm_model_prefix)

    def setup_model(self):
        config = self.config
        model_conf = config.model

        with UpdateConfig(model_conf):
            model_conf.input_dim = self.collate_fn_test.feature_size
            model_conf.output_dim = self.collate_fn_test.vocab_size

        model = U2Model.from_config(model_conf)

        if self.parallel:
            model = paddle.DataParallel(model)

        logger.info(f"{model}")
        layer_tools.print_params(model, logger.info)

        self.model = model
        logger.info("Setup model")

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def test(self):
        self.model.eval()
        cfg = self.config.decoding
        audio_file = self.audio_file
        collate_fn_test = self.collate_fn_test
        audio, _ = collate_fn_test.process_utterance(
            audio_file=audio_file, transcript="Hello")
        audio_len = audio.shape[0]
        audio = paddle.to_tensor(audio, dtype='float32')
        audio_len = paddle.to_tensor(audio_len)
        audio = paddle.unsqueeze(audio, axis=0)
        vocab_list = collate_fn_test.vocab_list

        text_feature = self.collate_fn_test.text_feature
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
        logger.info("The result_transcripts: " + result_transcripts[0][0])

    def run_test(self):
        self.resume()
        try:
            self.test()
        except KeyboardInterrupt:
            sys.exit(-1)

    def setup(self):
        """Setup the experiment.
        """
        paddle.set_device('gpu' if self.args.nprocs > 0 else 'cpu')

        #self.setup_output_dir()
        #self.setup_checkpointer()

        #self.setup_dataloader()
        self.setup_model()

        self.iteration = 0
        self.epoch = 0

    def resume(self):
        """Resume from the checkpoint at checkpoints in the output
        directory or load a specified checkpoint.
        """
        params_path = self.args.checkpoint_path + ".pdparams"
        model_dict = paddle.load(params_path)
        self.model.set_state_dict(model_dict)


def check(audio_file):
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


def main_sp(config, args):
    exp = U2Tester_Hub(config, args)
    with exp.eval():
        exp.setup()
        exp.run_test()


def main(config, args):
    main_sp(config, args)


if __name__ == "__main__":
    parser = default_argument_parser()
    # save asr result to
    parser.add_argument(
        "--result_file", type=str, help="path of save the asr result")
    parser.add_argument(
        "--audio_file", type=str, help="path of the input audio file")
    args = parser.parse_args()
    print_arguments(args, globals())

    if not os.path.isfile(args.audio_file):
        print("Please input the right audio file path")
        sys.exit(-1)
    check(args.audio_file)
    # https://yaml.org/type/float.html
    config = get_cfg_defaults()
    if args.config:
        config.merge_from_file(args.config)
    if args.opts:
        config.merge_from_list(args.opts)
    config.freeze()
    print(config)
    if args.dump_config:
        with open(args.dump_config, 'w') as f:
            print(config, file=f)

    # Setting for profiling
    pr = cProfile.Profile()
    pr.runcall(main, config, args)
    pr.dump_stats('test.profile')
