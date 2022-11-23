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
"""Server-end for the ASR demo."""
import functools

import numpy as np
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor
from paddle.io import DataLoader
from yacs.config import CfgNode

from paddlespeech.s2t.io.collator import SpeechCollator
from paddlespeech.s2t.io.dataset import ManifestDataset
from paddlespeech.s2t.models.ds2 import DeepSpeech2Model
from paddlespeech.s2t.training.cli import default_argument_parser
from paddlespeech.s2t.utils.socket_server import AsrRequestHandler
from paddlespeech.s2t.utils.socket_server import AsrTCPServer
from paddlespeech.s2t.utils.socket_server import warm_up_test
from paddlespeech.s2t.utils.utility import add_arguments
from paddlespeech.s2t.utils.utility import print_arguments


def init_predictor(args):
    if args.model_dir is not None:
        config = Config(args.model_dir)
    else:
        config = Config(args.model_file, args.params_file)

    config.enable_memory_optim()
    if args.use_gpu:
        config.enable_use_gpu(memory_pool_init_size_mb=1000, device_id=0)
    else:
        # If not specific mkldnn, you can set the blas thread.
        # The thread num should not be greater than the number of cores in the CPU.
        config.set_cpu_math_library_num_threads(4)
        config.enable_mkldnn()

    predictor = create_predictor(config)
    return predictor


def run(predictor, img):
    # copy img data to input tensor
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        #input_tensor.reshape(img[i].shape)
        #input_tensor.copy_from_cpu(img[i].copy())

    # do the inference
    predictor.run()

    results = []
    # get out data from output tensor
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)

    return results


def inference(config, args):
    predictor = init_predictor(args)


def start_server(config, args):
    """Start the ASR server"""
    config.defrost()
    config.manifest = config.test_manifest
    dataset = ManifestDataset.from_config(config)

    config.augmentation_config = ""
    config.keep_transcription_text = True
    config.batch_size = 1
    config.num_workers = 0
    collate_fn = SpeechCollator.from_config(config)
    test_loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=0)

    model = DeepSpeech2Model.from_pretrained(test_loader, config,
                                             args.checkpoint_path)
    model.eval()

    # prepare ASR inference handler
    def file_to_transcript(filename):
        feature = test_loader.collate_fn.process_utterance(filename, "")
        audio = np.array([feature[0]]).astype('float32')  #[1, T, D]
        audio_len = feature[0].shape[0]
        audio_len = np.array([audio_len]).astype('int64')  # [1]

        result_transcript = model.decode(
            paddle.to_tensor(audio),
            paddle.to_tensor(audio_len),
            vocab_list=test_loader.collate_fn.vocab_list,
            decoding_method=config.decode.decoding_method,
            lang_model_path=config.decode.lang_model_path,
            beam_alpha=config.decode.alpha,
            beam_beta=config.decode.beta,
            beam_size=config.decode.beam_size,
            cutoff_prob=config.decode.cutoff_prob,
            cutoff_top_n=config.decode.cutoff_top_n,
            num_processes=config.decode.num_proc_bsearch)
        return result_transcript[0]

    # warming up with utterrances sampled from Librispeech
    print('-----------------------------------------------------------')
    print('Warming up ...')
    warm_up_test(
        audio_process_handler=file_to_transcript,
        manifest_path=args.warmup_manifest,
        num_test_cases=3)
    print('-----------------------------------------------------------')

    # start the server
    server = AsrTCPServer(
        server_address=(args.host_ip, args.host_port),
        RequestHandlerClass=AsrRequestHandler,
        speech_save_dir=args.speech_save_dir,
        audio_process_handler=file_to_transcript)
    print("ASR Server Started.")
    server.serve_forever()


def main(config, args):
    start_server(config, args)


if __name__ == "__main__":
    parser = default_argument_parser()
    add_arg = functools.partial(add_arguments, argparser=parser)
    # yapf: disable
    add_arg('host_ip',          str,
            'localhost',
            "Server's IP address.")
    add_arg('host_port',        int,    8089,    "Server's IP port.")
    add_arg('speech_save_dir',  str,
            'demo_cache',
            "Directory to save demo audios.")
    add_arg('warmup_manifest',  str, None, "Filepath of manifest to warm up.")
    add_arg(
        "--model_file",
        type=str,
        default="",
        help="Model filename, Specify this when your model is a combined model."
    )
    add_arg(
        "--params_file",
        type=str,
        default="",
        help="Parameter filename, Specify this when your model is a combined model."
    )
    add_arg(
        "--model_dir",
        type=str,
        default=None,
        help="Model dir, If you load a non-combined model, specify the directory of the model."
    )
    add_arg("--use_gpu",
                        type=bool,
                        default=False,
                        help="Whether use gpu.")
    args = parser.parse_args()
    print_arguments(args, globals())

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

    args.warmup_manifest = config.test_manifest
    print_arguments(args, globals())

    if args.dump_config:
        with open(args.dump_config, 'w') as f:
            print(config, file=f)

    main(config, args)
