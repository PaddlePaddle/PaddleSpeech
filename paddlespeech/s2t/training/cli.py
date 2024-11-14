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

import distutils
from yacs.config import CfgNode

from paddlespeech.utils.argparse import strtobool


class ExtendAction(argparse.Action):
    """
    [Since Python 3.8, the "extend" is available directly in stdlib]
    (https://docs.python.org/3.8/library/argparse.html#action).
    If you only have to support 3.8+ then defining it yourself is no longer required. 
    Usage of stdlib "extend" action is exactly the same way as this answer originally described:
    """

    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest) or []
        items.extend(values)
        setattr(namespace, self.dest, items)


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            # parse arguments in the file and store them in the target namespace
            parser.parse_args(f.read().split(), namespace)


def default_argument_parser(parser=None):
    r"""A simple yet genral argument parser for experiments with t2s.

    This is used in examples with t2s. And it is intended to be used by
    other experiments with t2s. It requires a minimal set of command line
    arguments to start a training script.

    The ``--config`` and ``--opts`` are used for overwrite the deault
    configuration.

    The ``--data`` and ``--output`` specifies the data path and output path.
    Resuming training from existing progress at the output directory is the
    intended default behavior.

    The ``--checkpoint_path`` specifies the checkpoint to load from.

    The ``--ngpu`` specifies how to run the training.


    See Also
    --------
    paddlespeech.t2s.training.experiment
    Returns
    -------
    argparse.ArgumentParser
        the parser
    """
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.register('action', 'extend', ExtendAction)
    parser.add_argument(
        '--conf', type=open, action=LoadFromFile, help="config file.")
    parser.add_argument(
        "--debug",
        type=strtobool,
        default=False,
        help="logging with debug mode.")
    parser.add_argument(
        "--dump_path", type=str, default=None, help="path to dump config file.")

    # train group
    train_group = parser.add_argument_group(
        title='Train Options', description=None)
    train_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed to use for paddle, np and random. None or 0 for random, else set seed."
    )
    train_group.add_argument(
        "--ngpu",
        type=int,
        default=1,
        help="number of parallel processes. 0 for cpu.")
    train_group.add_argument(
        '--nxpu',
        type=int,
        default=0,
        choices=[0, 1],
        help="if nxpu == 0 and ngpu == 0, use cpu.")
    train_group.add_argument(
        "--config", metavar="CONFIG_FILE", help="config file.")
    train_group.add_argument(
        "--output", metavar="CKPT_DIR", help="path to save checkpoint.")
    train_group.add_argument(
        "--checkpoint_path", type=str, help="path to load checkpoint")
    train_group.add_argument(
        "--opts",
        action='extend',
        nargs=2,
        metavar=('key', 'val'),
        help="overwrite --config field, passing (KEY VALUE) pairs")
    train_group.add_argument(
        "--dump-config", metavar="FILE", help="dump config to `this` file.")

    # test group
    test_group = parser.add_argument_group(
        title='Test Options', description=None)
    test_group.add_argument(
        "--decode_cfg",
        metavar="DECODE_CONFIG_FILE",
        help="decode config file.")
    test_group.add_argument(
        "--result_file", type=str, help="path of save the asr result")
    test_group.add_argument(
        "--audio_file", type=str, help="path of the input audio file")

    # quant & export
    quant_group = parser.add_argument_group(
        title='Quant Options', description=None)
    quant_group.add_argument(
        "--audio_scp", type=str, help="path of the input audio scp file")
    quant_group.add_argument(
        "--num_utts",
        type=int,
        default=200,
        help="num utts for quant calibrition.")
    quant_group.add_argument(
        "--export_path",
        type=str,
        default='export.jit.quant',
        help="path of the jit model to save")

    # profile group
    profile_group = parser.add_argument_group(
        title='Benchmark Options', description=None)
    profile_group.add_argument(
        '--profiler-options',
        type=str,
        default=None,
        help='The option of profiler, which should be in format \"key1=value1;key2=value2;key3=value3\".'
    )
    profile_group.add_argument(
        '--benchmark-batch-size',
        type=int,
        default=None,
        help='batch size for benchmark.')
    profile_group.add_argument(
        '--benchmark-max-step',
        type=int,
        default=None,
        help='max iteration for benchmark.')

    return parser


def config_from_args(args):
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
    return config


def maybe_dump_config(dump_path, config):
    if dump_path:
        with open(dump_path, 'w') as f:
            print(config, file=f)
        print(f"save config to {dump_path}")
