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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.∏
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from Whisper (https://github.com/openai/whisper/whisper/)
import os.path
import sys

import distutils
import numpy as np
import paddle
import soundfile
from yacs.config import CfgNode

from paddlespeech.s2t.models.whisper import log_mel_spectrogram
from paddlespeech.s2t.models.whisper import ModelDimensions
from paddlespeech.s2t.models.whisper import transcribe
from paddlespeech.s2t.models.whisper import Whisper
from paddlespeech.s2t.training.cli import default_argument_parser
from paddlespeech.s2t.utils.log import Log
from paddlespeech.utils.argparse import strtobool

logger = Log(__name__).getlog()


class WhisperInfer():
    def __init__(self, config, args):
        self.args = args
        self.config = config
        self.audio_file = args.audio_file

        paddle.set_device('gpu' if self.args.ngpu > 0 else 'cpu')
        config.pop("ngpu")

        #load_model
        model_dict = paddle.load(self.config.model_file)
        config.pop("model_file")
        dims = ModelDimensions(**model_dict["dims"])
        self.model = Whisper(dims)
        self.model.load_dict(model_dict)

    def run(self):
        check(args.audio_file)

        with paddle.no_grad():
            temperature = config.pop("temperature")
            temperature_increment_on_fallback = config.pop(
                "temperature_increment_on_fallback")
            if temperature_increment_on_fallback is not None:
                temperature = tuple(
                    np.arange(temperature, 1.0 + 1e-6,
                              temperature_increment_on_fallback))
            else:
                temperature = [temperature]

            #load audio
            mel = log_mel_spectrogram(
                args.audio_file, resource_path=config.resource_path)

            result = transcribe(
                self.model, mel, temperature=temperature, **config)
            if args.result_file is not None:
                with open(args.result_file, 'w') as f:
                    f.write(str(result))
            return result


def check(audio_file: str):
    if not os.path.isfile(audio_file):
        print("Please input the right audio file path")
        sys.exit(-1)

    logger.info("checking the audio file format......")
    try:
        _, sample_rate = soundfile.read(audio_file)
    except Exception as e:
        logger.error(str(e))
        logger.error(
            "can not open the wav file, please check the audio file format")
        sys.exit(-1)
    logger.info("The sample rate is %d" % sample_rate)
    assert (sample_rate == 16000)
    logger.info("The audio file format is right")


def main(config, args):
    WhisperInfer(config, args).run()


if __name__ == "__main__":
    parser = default_argument_parser()
    # save asr result to
    parser.add_argument(
        "--result_file", type=str, help="path of save the asr result")
    parser.add_argument(
        "--audio_file", type=str, help="path of the input audio file")
    parser.add_argument(
        "--debug", type=strtobool, default=False, help="for debug.")
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
