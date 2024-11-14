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
# Reference espnet Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
# Modified from espnet(https://github.com/espnet/espnet)
"""End-to-end speech recognition model decoding script."""
import logging
import os
import random
import sys

import configargparse
import numpy as np

from paddlespeech.utils.argparse import strtobool


def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transcribe text from speech using "
        "a speech recognition model on one CPU or GPU",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter, )
    parser.add(
        '--model-name',
        type=str,
        default='u2_kaldi',
        help='model name, e.g: deepspeech2, u2, u2_kaldi, u2_st')
    # general configuration
    parser.add("--config", is_config_file=True, help="Config file path")
    parser.add(
        "--config2",
        is_config_file=True,
        help="Second config file path that overwrites the settings in `--config`",
    )
    parser.add(
        "--config3",
        is_config_file=True,
        help="Third config file path that overwrites the settings "
        "in `--config` and `--config2`", )

    parser.add_argument("--ngpu", type=int, default=0, help="Number of GPUs")
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32", "float64"),
        default="float32",
        help="Float precision (only available in --api v2)", )
    parser.add_argument("--debugmode", type=int, default=1, help="Debugmode")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--verbose", "-V", type=int, default=2, help="Verbose option")
    parser.add_argument(
        "--batchsize",
        type=int,
        default=1,
        help="Batch size for beam search (0: means no batch processing)", )
    parser.add_argument(
        "--preprocess-conf",
        type=str,
        default=None,
        help="The configuration file for the pre-processing", )
    parser.add_argument(
        "--api",
        default="v2",
        choices=["v2"],
        help="Beam search APIs "
        "v2: Experimental API. It supports any models that implements ScorerInterface.",
    )
    # task related
    parser.add_argument(
        "--recog-json", type=str, help="Filename of recognition data (json)")
    parser.add_argument(
        "--result-label",
        type=str,
        required=True,
        help="Filename of result label data (json)", )
    # model (parameter) related
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model file parameters to read")
    parser.add_argument(
        "--model-conf", type=str, default=None, help="Model config file")
    parser.add_argument(
        "--num-spkrs",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of speakers in the speech", )
    parser.add_argument(
        "--num-encs",
        default=1,
        type=int,
        help="Number of encoders in the model.")
    # search related
    parser.add_argument(
        "--nbest", type=int, default=1, help="Output N-best hypotheses")
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size")
    parser.add_argument(
        "--penalty", type=float, default=0.0, help="Incertion penalty")
    parser.add_argument(
        "--maxlenratio",
        type=float,
        default=0.0,
        help="""Input length ratio to obtain max output length.
                        If maxlenratio=0.0 (default), it uses a end-detect function
                        to automatically find maximum hypothesis lengths.
                        If maxlenratio<0.0, its absolute value is interpreted
                        as a constant max output length""", )
    parser.add_argument(
        "--minlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain min output length", )
    parser.add_argument(
        "--ctc-weight",
        type=float,
        default=0.0,
        help="CTC weight in joint decoding")
    parser.add_argument(
        "--weights-ctc-dec",
        type=float,
        action="append",
        help="ctc weight assigned to each encoder during decoding."
        "[in multi-encoder mode only]", )
    parser.add_argument(
        "--ctc-window-margin",
        type=int,
        default=0,
        help="""Use CTC window with margin parameter to accelerate
                        CTC/attention decoding especially on GPU. Smaller magin
                        makes decoding faster, but may increase search errors.
                        If margin=0 (default), this function is disabled""", )
    # transducer related
    parser.add_argument(
        "--search-type",
        type=str,
        default="default",
        choices=["default", "nsc", "tsd", "alsd", "maes"],
        help="""Type of beam search implementation to use during inference.
        Can be either: default beam search ("default"),
        N-Step Constrained beam search ("nsc"), Time-Synchronous Decoding ("tsd"),
        Alignment-Length Synchronous Decoding ("alsd") or
        modified Adaptive Expansion Search ("maes").""", )
    parser.add_argument(
        "--nstep",
        type=int,
        default=1,
        help="""Number of expansion steps allowed in NSC beam search or mAES
        (nstep > 0 for NSC and nstep > 1 for mAES).""", )
    parser.add_argument(
        "--prefix-alpha",
        type=int,
        default=2,
        help="Length prefix difference allowed in NSC beam search or mAES.", )
    parser.add_argument(
        "--max-sym-exp",
        type=int,
        default=2,
        help="Number of symbol expansions allowed in TSD.", )
    parser.add_argument(
        "--u-max",
        type=int,
        default=400,
        help="Length prefix difference allowed in ALSD.", )
    parser.add_argument(
        "--expansion-gamma",
        type=float,
        default=2.3,
        help="Allowed logp difference for prune-by-value method in mAES.", )
    parser.add_argument(
        "--expansion-beta",
        type=int,
        default=2,
        help="""Number of additional candidates for expanded hypotheses
                selection in mAES.""", )
    parser.add_argument(
        "--score-norm",
        type=strtobool,
        nargs="?",
        default=True,
        help="Normalize final hypotheses' score by length", )
    parser.add_argument(
        "--softmax-temperature",
        type=float,
        default=1.0,
        help="Penalization term for softmax function.", )
    # rnnlm related
    parser.add_argument(
        "--rnnlm", type=str, default=None, help="RNNLM model file to read")
    parser.add_argument(
        "--rnnlm-conf",
        type=str,
        default=None,
        help="RNNLM model config file to read")
    parser.add_argument(
        "--word-rnnlm",
        type=str,
        default=None,
        help="Word RNNLM model file to read")
    parser.add_argument(
        "--word-rnnlm-conf",
        type=str,
        default=None,
        help="Word RNNLM model config file to read", )
    parser.add_argument(
        "--word-dict", type=str, default=None, help="Word list to read")
    parser.add_argument(
        "--lm-weight", type=float, default=0.1, help="RNNLM weight")
    # ngram related
    parser.add_argument(
        "--ngram-model",
        type=str,
        default=None,
        help="ngram model file to read")
    parser.add_argument(
        "--ngram-weight", type=float, default=0.1, help="ngram weight")
    parser.add_argument(
        "--ngram-scorer",
        type=str,
        default="part",
        choices=("full", "part"),
        help="""if the ngram is set as a part scorer, similar with CTC scorer,
                ngram scorer only scores topK hypethesis.
                if the ngram is set as full scorer, ngram scorer scores all hypthesis
                the decoding speed of part scorer is musch faster than full one""",
    )
    # streaming related
    parser.add_argument(
        "--streaming-mode",
        type=str,
        default=None,
        choices=["window", "segment"],
        help="""Use streaming recognizer for inference.
                        `--batchsize` must be set to 0 to enable this mode""", )
    parser.add_argument(
        "--streaming-window", type=int, default=10, help="Window size")
    parser.add_argument(
        "--streaming-min-blank-dur",
        type=int,
        default=10,
        help="Minimum blank duration threshold", )
    parser.add_argument(
        "--streaming-onset-margin", type=int, default=1, help="Onset margin")
    parser.add_argument(
        "--streaming-offset-margin", type=int, default=1, help="Offset margin")
    # non-autoregressive related
    # Mask CTC related. See https://arxiv.org/abs/2005.08700 for the detail.
    parser.add_argument(
        "--maskctc-n-iterations",
        type=int,
        default=10,
        help="Number of decoding iterations."
        "For Mask CTC, set 0 to predict 1 mask/iter.", )
    parser.add_argument(
        "--maskctc-probability-threshold",
        type=float,
        default=0.999,
        help="Threshold probability for CTC output", )
    # quantize model related
    parser.add_argument(
        "--quantize-config",
        nargs="*",
        help="Quantize config list. E.g.: --quantize-config=[Linear,LSTM,GRU]",
    )
    parser.add_argument(
        "--quantize-dtype",
        type=str,
        default="qint8",
        help="Dtype dynamic quantize")
    parser.add_argument(
        "--quantize-asr-model",
        type=bool,
        default=False,
        help="Quantize asr model", )
    parser.add_argument(
        "--quantize-lm-model",
        type=bool,
        default=False,
        help="Quantize lm model", )
    return parser


def main(args):
    """Run the main decoding function."""
    parser = get_parser()
    parser.add_argument(
        "--output", metavar="CKPT_DIR", help="path to save checkpoint.")
    parser.add_argument(
        "--checkpoint_path", type=str, help="path to load checkpoint")
    parser.add_argument("--dict-path", type=str, help="path to load checkpoint")
    args = parser.parse_args(args)

    if args.ngpu == 0 and args.dtype == "float16":
        raise ValueError(
            f"--dtype {args.dtype} does not support the CPU backend.")

    # logging info
    if args.verbose == 1:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose == 2:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")
    logging.info(args)

    # check CUDA_VISIBLE_DEVICES
    if args.ngpu > 0:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
        elif args.ngpu != len(cvd.split(",")):
            logging.error("#gpus is not matched with CUDA_VISIBLE_DEVICES.")
            sys.exit(1)

        # TODO(mn5k): support of multiple GPUs
        if args.ngpu > 1:
            logging.error("The program only supports ngpu=1.")
            sys.exit(1)

    # display PYTHONPATH
    logging.info("python path = " + os.environ.get("PYTHONPATH", "(None)"))

    # seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    logging.info("set random seed = %d" % args.seed)

    # validate rnn options
    if args.rnnlm is not None and args.word_rnnlm is not None:
        logging.error(
            "It seems that both --rnnlm and --word-rnnlm are specified. "
            "Please use either option.")
        sys.exit(1)

    # recog
    if args.num_spkrs == 1:
        if args.num_encs == 1:
            # Experimental API that supports custom LMs
            if args.api == "v2":
                from paddlespeech.s2t.decoders.recog import recog_v2
                recog_v2(args)
            else:
                raise ValueError("Only support --api v2")
        else:
            if args.api == "v2":
                raise NotImplementedError(
                    f"--num-encs {args.num_encs} > 1 is not supported in --api v2"
                )
    elif args.num_spkrs == 2:
        raise ValueError("asr_mix not supported.")


if __name__ == "__main__":
    main(sys.argv[1:])
