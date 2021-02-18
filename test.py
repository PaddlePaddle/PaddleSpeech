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

import io
import logging
import argparse
import functools

from paddle import distributed as dist

from utils.utility import print_arguments
from training.cli import default_argument_parser

from model_utils.config import get_cfg_defaults
from model_utils.model import DeepSpeech2Trainer as Trainer
from utils.error_rate import char_errors, word_errors


def evaluate():
    """Evaluate on whole test data for DeepSpeech2."""

    # decoders only accept string encoded in utf-8
    vocab_list = [chars for chars in data_generator.vocab_list]

    errors_func = char_errors if args.error_rate_type == 'cer' else word_errors
    errors_sum, len_refs, num_ins = 0.0, 0, 0
    ds2_model.logger.info("start evaluation ...")

    for infer_data in batch_reader():
        probs_split = ds2_model.infer_batch_probs(
            infer_data=infer_data, feeding_dict=data_generator.feeding)

        if args.decoding_method == "ctc_greedy":
            result_transcripts = ds2_model.decode_batch_greedy(
                probs_split=probs_split, vocab_list=vocab_list)
        else:
            result_transcripts = ds2_model.decode_batch_beam_search(
                probs_split=probs_split,
                beam_alpha=args.alpha,
                beam_beta=args.beta,
                beam_size=args.beam_size,
                cutoff_prob=args.cutoff_prob,
                cutoff_top_n=args.cutoff_top_n,
                vocab_list=vocab_list,
                num_processes=args.num_proc_bsearch)

        target_transcripts = infer_data[1]

        for target, result in zip(target_transcripts, result_transcripts):
            errors, len_ref = errors_func(target, result)
            errors_sum += errors
            len_refs += len_ref
            num_ins += 1
        print("Error rate [%s] (%d/?) = %f" %
              (args.error_rate_type, num_ins, errors_sum / len_refs))

    print("Final error rate [%s] (%d/%d) = %f" %
          (args.error_rate_type, num_ins, num_ins, errors_sum / len_refs))

    ds2_model.logger.info("finish evaluation")


def main_sp(config, args):
    exp = Trainer(config, args)
    exp.setup()
    exp.run()


def main(config, args):
    main_sp(config, args)


if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    print_arguments(args)

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

    main(config, args)
