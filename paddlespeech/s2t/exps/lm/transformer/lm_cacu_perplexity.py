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
# Caculating the PPL of LM model
import os

import numpy as np
import paddle
from paddle.io import DataLoader
from yacs.config import CfgNode

from paddlespeech.s2t.models.lm.dataset import TextCollatorSpm
from paddlespeech.s2t.models.lm.dataset import TextDataset
from paddlespeech.s2t.models.lm_interface import dynamic_import_lm
from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()


def get_config(config_path):
    confs = CfgNode(new_allowed=True)
    confs.merge_from_file(config_path)
    return confs


def load_trained_lm(args):
    lm_config = get_config(args.rnnlm_conf)
    lm_model_module = lm_config.model_module
    lm_class = dynamic_import_lm(lm_model_module)
    lm = lm_class(**lm_config.model)
    model_dict = paddle.load(args.rnnlm)
    lm.set_state_dict(model_dict)
    return lm, lm_config


def write_dict_into_file(ppl_dict, name):
    with open(name, "w") as f:
        for key in ppl_dict.keys():
            f.write(key + " " + ppl_dict[key] + "\n")
    return


def cacu_perplexity(
        lm_model,
        lm_config,
        args,
        log_base=None, ):
    unit_type = lm_config.data.unit_type
    batch_size = lm_config.decoding.batch_size
    num_workers = lm_config.decoding.num_workers
    text_file_path = args.text_path

    total_nll = 0.0
    total_ntokens = 0
    ppl_dict = {}
    len_dict = {}
    text_dataset = TextDataset.from_file(text_file_path)
    collate_fn_text = TextCollatorSpm(
        unit_type=unit_type,
        vocab_filepath=args.vocab_path,
        spm_model_prefix=args.bpeprefix)
    train_loader = DataLoader(
        text_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_text,
        num_workers=num_workers)

    logger.info("start caculating PPL......")
    for i, (keys, ys_input_pad, ys_output_pad,
            y_lens) in enumerate(train_loader()):

        ys_input_pad = paddle.to_tensor(ys_input_pad)
        ys_output_pad = paddle.to_tensor(ys_output_pad)
        _, unused_logp, unused_count, nll, nll_count = lm_model.forward(
            ys_input_pad, ys_output_pad)
        nll = nll.numpy()
        nll_count = nll_count.numpy()
        for key, _nll, ntoken in zip(keys, nll, nll_count):
            if log_base is None:
                utt_ppl = np.exp(_nll / ntoken)
            else:
                utt_ppl = log_base**(_nll / ntoken / np.log(log_base))

            # Write PPL of each utts for debugging or analysis
            ppl_dict[key] = str(utt_ppl)
            len_dict[key] = str(ntoken)

        total_nll += nll.sum()
        total_ntokens += nll_count.sum()
        logger.info("Current total nll: " + str(total_nll))
        logger.info("Current total tokens: " + str(total_ntokens))
    write_dict_into_file(ppl_dict, os.path.join(args.output_dir, "uttPPL"))
    write_dict_into_file(len_dict, os.path.join(args.output_dir, "uttLEN"))
    if log_base is None:
        ppl = np.exp(total_nll / total_ntokens)
    else:
        ppl = log_base**(total_nll / total_ntokens / np.log(log_base))

    if log_base is None:
        log_base = np.e
    else:
        log_base = log_base

    return ppl, log_base


def run_get_perplexity(args):
    if args.ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
    if args.ngpu == 1:
        device = "gpu:0"
    else:
        device = "cpu"
    paddle.set_device(device)
    dtype = getattr(paddle, args.dtype)
    logger.info(f"Decoding device={device}, dtype={dtype}")
    lm_model, lm_config = load_trained_lm(args)
    lm_model.to(device=device, dtype=dtype)
    lm_model.eval()
    PPL, log_base = cacu_perplexity(lm_model, lm_config, args, None)
    logger.info("Final PPL: " + str(PPL))
    logger.info("The log base is:" + str("%.2f" % log_base))
