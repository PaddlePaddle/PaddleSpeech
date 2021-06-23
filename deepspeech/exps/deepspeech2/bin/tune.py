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
"""Beam search parameters tuning for DeepSpeech2 model."""
import functools
import sys

import numpy as np
from paddle.io import DataLoader

from deepspeech.exps.deepspeech2.config import get_cfg_defaults
from deepspeech.io.collator import SpeechCollator
from deepspeech.io.dataset import ManifestDataset
from deepspeech.models.deepspeech2 import DeepSpeech2Model
from deepspeech.training.cli import default_argument_parser
from deepspeech.utils import error_rate
from deepspeech.utils.utility import add_arguments
from deepspeech.utils.utility import print_arguments


def tune(config, args):
    """Tune parameters alpha and beta incrementally."""
    if not args.num_alphas >= 0:
        raise ValueError("num_alphas must be non-negative!")
    if not args.num_betas >= 0:
        raise ValueError("num_betas must be non-negative!")
    config.defrost()
    config.data.manfiest = config.data.dev_manifest
    config.data.augmentation_config = ""
    config.data.keep_transcription_text = True
    dev_dataset = ManifestDataset.from_config(config)

    valid_loader = DataLoader(
        dev_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=SpeechCollator(keep_transcription_text=True))

    model = DeepSpeech2Model.from_pretrained(valid_loader, config,
                                             args.checkpoint_path)
    model.eval()

    # decoders only accept string encoded in utf-8
    vocab_list = valid_loader.dataset.vocab_list
    errors_func = error_rate.char_errors if config.decoding.error_rate_type == 'cer' else error_rate.word_errors

    # create grid for search
    cand_alphas = np.linspace(args.alpha_from, args.alpha_to, args.num_alphas)
    cand_betas = np.linspace(args.beta_from, args.beta_to, args.num_betas)
    params_grid = [(alpha, beta) for alpha in cand_alphas
                   for beta in cand_betas]

    err_sum = [0.0 for i in range(len(params_grid))]
    err_ave = [0.0 for i in range(len(params_grid))]

    num_ins, len_refs, cur_batch = 0, 0, 0
    # initialize external scorer
    model.decoder.init_decode(args.alpha_from, args.beta_from,
                              config.decoding.lang_model_path, vocab_list,
                              config.decoding.decoding_method)
    ## incremental tuning parameters over multiple batches
    print("start tuning ...")
    for infer_data in valid_loader():
        if (args.num_batches >= 0) and (cur_batch >= args.num_batches):
            break

        def ordid2token(texts, texts_len):
            """ ord() id to chr() chr """
            trans = []
            for text, n in zip(texts, texts_len):
                n = n.numpy().item()
                ids = text[:n]
                trans.append(''.join([chr(i) for i in ids]))
            return trans

        audio, audio_len, text, text_len = infer_data
        target_transcripts = ordid2token(text, text_len)
        num_ins += audio.shape[0]

        # model infer
        eouts, eouts_len = model.encoder(audio, audio_len)
        probs = model.decoder.softmax(eouts)

        # grid search
        for index, (alpha, beta) in enumerate(params_grid):
            print(f"tuneing: alpha={alpha} beta={beta}")
            result_transcripts = model.decoder.decode_probs(
                probs.numpy(), eouts_len, vocab_list,
                config.decoding.decoding_method,
                config.decoding.lang_model_path, alpha, beta,
                config.decoding.beam_size, config.decoding.cutoff_prob,
                config.decoding.cutoff_top_n, config.decoding.num_proc_bsearch)

            for target, result in zip(target_transcripts, result_transcripts):
                errors, len_ref = errors_func(target, result)
                err_sum[index] += errors

                # accumulate the length of references of every batchπ
                # in the first iteration
                if args.alpha_from == alpha and args.beta_from == beta:
                    len_refs += len_ref

            err_ave[index] = err_sum[index] / len_refs
            if index % 2 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
            print("tuneing: one grid done!")

        # output on-line tuning result at the end of current batch
        err_ave_min = min(err_ave)
        min_index = err_ave.index(err_ave_min)
        print("\nBatch %d [%d/?], current opt (alpha, beta) = (%s, %s), "
              " min [%s] = %f" %
              (cur_batch, num_ins, "%.3f" % params_grid[min_index][0],
               "%.3f" % params_grid[min_index][1],
               config.decoding.error_rate_type, err_ave_min))
        cur_batch += 1

    # output WER/CER at every (alpha, beta)
    print("\nFinal %s:\n" % config.decoding.error_rate_type)
    for index in range(len(params_grid)):
        print("(alpha, beta) = (%s, %s), [%s] = %f" %
              ("%.3f" % params_grid[index][0], "%.3f" % params_grid[index][1],
               config.decoding.error_rate_type, err_ave[index]))

    err_ave_min = min(err_ave)
    min_index = err_ave.index(err_ave_min)
    print("\nFinish tuning on %d batches, final opt (alpha, beta) = (%s, %s)" %
          (cur_batch, "%.3f" % params_grid[min_index][0],
           "%.3f" % params_grid[min_index][1]))

    print("finish tuning")


def main(config, args):
    tune(config, args)


if __name__ == "__main__":
    parser = default_argument_parser()
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('num_batches', int, -1, "# of batches tuning on. "
            "Default -1, on whole dev set.")
    add_arg('num_alphas', int, 45, "# of alpha candidates for tuning.")
    add_arg('num_betas', int, 8, "# of beta candidates for tuning.")
    add_arg('alpha_from', float, 1.0, "Where alpha starts tuning from.")
    add_arg('alpha_to', float, 3.2, "Where alpha ends tuning with.")
    add_arg('beta_from', float, 0.1, "Where beta starts tuning from.")
    add_arg('beta_to', float, 0.45, "Where beta ends tuning with.")

    add_arg('batch_size', int, 256, "# of samples per batch.")
    add_arg('beam_size', int, 500, "Beam search width.")
    add_arg('num_proc_bsearch', int, 8, "# of CPUs for beam search.")
    add_arg('cutoff_prob', float, 1.0, "Cutoff probability for pruning.")
    add_arg('cutoff_top_n', int, 40, "Cutoff number for pruning.")

    args = parser.parse_args()
    print_arguments(args, globals())

    # https://yaml.org/type/float.html
    config = get_cfg_defaults()
    if args.config:
        config.merge_from_file(args.config)
    if args.opts:
        config.merge_from_list(args.opts)

    config.data.batch_size = args.batch_size
    config.decoding.beam_size = args.beam_size
    config.decoding.num_proc_bsearch = args.num_proc_bsearch
    config.decoding.cutoff_prob = args.cutoff_prob
    config.decoding.cutoff_top_n = args.cutoff_top_n

    config.freeze()
    print(config)

    if args.dump_config:
        with open(args.dump_config, 'w') as f:
            print(config, file=f)

    main(config, args)
