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


def default_argument_parser():
    r"""A simple yet genral argument parser for experiments with parakeet.

    This is used in examples with parakeet. And it is intended to be used by
    other experiments with parakeet. It requires a minimal set of command line
    arguments to start a training script.

    The ``--config`` and ``--opts`` are used for overwrite the deault
    configuration.

    The ``--data`` and ``--output`` specifies the data path and output path.
    Resuming training from existing progress at the output directory is the
    intended default behavior.

    The ``--checkpoint_path`` specifies the checkpoint to load from.

    The ``--device`` and ``--nprocs`` specifies how to run the training.


    See Also
    --------
    parakeet.training.experiment
    Returns
    -------
    argparse.ArgumentParser
        the parser
    """
    parser = argparse.ArgumentParser()

    # yapf: disable
    train_group = parser.add_argument_group(title='Train Options', description=None)
    train_group.add_argument("--seed", type=int, default=None,
                        help="seed to use for paddle, np and random. None or 0 for random, else set seed.")
    train_group.add_argument("--device", type=str, default='gpu', choices=["cpu", "gpu"],
        help="device cpu and gpu are supported.")
    train_group.add_argument("--nprocs", type=int, default=1, help="number of parallel processes. 0 for cpu.")
    train_group.add_argument("--config", metavar="CONFIG_FILE", help="config file.")
    train_group.add_argument("--output", metavar="CKPT_DIR", help="path to save checkpoint.")
    train_group.add_argument("--checkpoint_path", type=str, help="path to load checkpoint")
    train_group.add_argument("--opts", type=str, default=[], nargs='+',
                        help="overwrite --config file, passing in LIST[KEY VALUE] pairs")
    train_group.add_argument("--dump-config", metavar="FILE", help="dump config to `this` file.")

    bech_group = parser.add_argument_group(title='Benchmark Options', description=None)
    bech_group.add_argument('--profiler-options', type=str, default=None,
        help='The option of profiler, which should be in format \"key1=value1;key2=value2;key3=value3\".')
    bech_group.add_argument('--benchmark-batch-size', type=int, default=None, help='batch size for benchmark.')
    bech_group.add_argument('--benchmark-max-step', type=int, default=None, help='max iteration for benchmark.')
    # yapd: enable

    return parser
