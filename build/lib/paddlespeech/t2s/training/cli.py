# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
    parser = argparse.ArgumentParser()

    # yapf: disable
    # data and outpu
    parser.add_argument("--config", metavar="FILE", help="path of the config file to overwrite to default config with.")
    parser.add_argument("--data", metavar="DATA_DIR", help="path to the datatset.")
    parser.add_argument("--output", metavar="OUTPUT_DIR", help="path to save checkpoint and logs.")

    # load from saved checkpoint
    parser.add_argument("--checkpoint_path", type=str, help="path of the checkpoint to load")

    # running
    parser.add_argument("--ngpu", type=int, default=1, help="if ngpu == 0, use cpu.")

    # overwrite extra config and default config
    parser.add_argument("--opts", nargs=argparse.REMAINDER, help="options to overwrite --config file and the default config, passing in KEY VALUE pairs")
    # yapd: enable

    return parser
