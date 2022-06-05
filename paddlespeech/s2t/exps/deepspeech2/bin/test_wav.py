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
"""Evaluation for DeepSpeech2 model."""
import os
import sys
from pathlib import Path

import paddle
import soundfile
from yacs.config import CfgNode

from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.io.collator import SpeechCollator
from paddlespeech.s2t.models.ds2 import DeepSpeech2Model
from paddlespeech.s2t.training.cli import default_argument_parser
from paddlespeech.s2t.utils import mp_tools
from paddlespeech.s2t.utils.checkpoint import Checkpoint
from paddlespeech.s2t.utils.log import Log
from paddlespeech.s2t.utils.utility import print_arguments
from paddlespeech.s2t.utils.utility import UpdateConfig

logger = Log(__name__).getlog()


class DeepSpeech2Tester_hub():
    def __init__(self, config, args):
        self.args = args
        self.config = config
        self.audio_file = args.audio_file
        self.collate_fn_test = SpeechCollator.from_config(config)
        self._text_featurizer = TextFeaturizer(
            unit_type=config.unit_type, vocab=None)

    def compute_result_transcripts(self, audio, audio_len, vocab_list, cfg):
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

        return result_transcripts

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def test(self):
        self.model.eval()
        cfg = self.config
        audio_file = self.audio_file
        collate_fn_test = self.collate_fn_test
        audio, _ = collate_fn_test.process_utterance(
            audio_file=audio_file, transcript=" ")
        audio_len = audio.shape[0]
        audio = paddle.to_tensor(audio, dtype='float32')
        audio_len = paddle.to_tensor(audio_len)
        audio = paddle.unsqueeze(audio, axis=0)
        vocab_list = collate_fn_test.vocab_list
        result_transcripts = self.compute_result_transcripts(
            audio, audio_len, vocab_list, cfg.decode)
        logger.info("result_transcripts: " + result_transcripts[0])

    def run_test(self):
        self.resume()
        try:
            self.test()
        except KeyboardInterrupt:
            exit(-1)

    def setup(self):
        """Setup the experiment.
        """
        paddle.set_device('gpu' if self.args.ngpu > 0 else 'cpu')

        self.setup_output_dir()
        self.setup_checkpointer()

        self.setup_model()

    def setup_output_dir(self):
        """Create a directory used for output.
        """
        # output dir
        if self.args.output:
            output_dir = Path(self.args.output).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(
                self.args.checkpoint_path).expanduser().parent.parent
            output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

    def setup_model(self):
        config = self.config.clone()
        with UpdateConfig(config):
            config.input_dim = self.collate_fn_test.feature_size
            config.output_dim = self.collate_fn_test.vocab_size

        model = DeepSpeech2Model.from_config(config)

        self.model = model

    def setup_checkpointer(self):
        """Create a directory used to save checkpoints into.

        It is "checkpoints" inside the output directory.
        """
        # checkpoint dir
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        self.checkpoint_dir = checkpoint_dir

        self.checkpoint = Checkpoint(
            kbest_n=self.config.checkpoint.kbest_n,
            latest_n=self.config.checkpoint.latest_n)

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
    exp = DeepSpeech2Tester_hub(config, args)
    exp.setup()
    exp.run_test()


def main(config, args):
    main_sp(config, args)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--audio_file", type=str, help='audio file path')
    # save asr result to
    parser.add_argument(
        "--result_file", type=str, help="path of save the asr result")
    args = parser.parse_args()
    print_arguments(args, globals())
    if not os.path.isfile(args.audio_file):
        print("Please input the audio file path")
        sys.exit(-1)
    check(args.audio_file)

    # https://yaml.org/type/float.html
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
    print(config)
    if args.dump_config:
        with open(args.dump_config, 'w') as f:
            print(config, file=f)

    main(config, args)
