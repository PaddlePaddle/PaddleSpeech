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
import numpy as np
import soundfile
import paddle
from paddlespeech.audio import load
from paddlespeech.s2t.training.cli import default_argument_parser
from paddlespeech.s2t.utils.utility import print_arguments
from paddlespeech.s2t.utils.log import Log
from paddlespeech.s2t.models.whisper import _download, transcribe, utils, ModelDimensions, Whisper

logger = Log(__name__).getlog()

def load_model(model_file):
    logger.info("download and loading the model file......")
    download_root = os.getenv(
            "XDG_CACHE_HOME", 
            os.path.join(os.path.expanduser("~"), ".cache", "whisper")
        )
    model_file = _download(args.model_file, download_root, in_memory = False)
    model_dict = paddle.load(model_file)
    dims = ModelDimensions(**model_dict["dims"])
    model = Whisper(dims)
    model.load_dict(model_dict)
    return model

def check(audio_file : str):
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

if __name__ == "__main__":
    parser = default_argument_parser()

    parser.add_argument(
        "--result_file", type=str, help="path of save the asr result")
    parser.add_argument(
        "--audio_file", type=str, help="path of the input audio file")
    parser.add_argument(
        "--model_file", default="large", type=str, help="path of the input model file")
    parser.add_argument("--beam_size",type=utils.optional_int, default=5)
    parser.add_argument("--verbose", type=utils.str2bool, default=True)
    parser.add_argument("--device", default="gpu")

    args = parser.parse_args()
    
    check(args.audio_file)

    available_device = paddle.get_device()
    if args.device == "cpu" and "gpu:" in available_device:
        warnings.warn("Performing inference on CPU when CUDA is available")
        paddle.set_device("cpu")
    else:
        paddle.set_device("gpu")
    
    model = load_model(args.model_file)

    result = transcribe(model, args.audio_file, beam_size = args.beam_size, fp16 = False, verbose = True)
