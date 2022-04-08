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
"""V2 backend for `asr_recog.py` using py:class:`decoders.beam_search.BeamSearch`."""
import jsonlines
import paddle
from yacs.config import CfgNode

from .beam_search import BatchBeamSearch
from .beam_search import BeamSearch
from .scorers.length_bonus import LengthBonus
from .scorers.scorer_interface import BatchScorerInterface
from .utils import add_results_to_json
from paddlespeech.s2t.exps import dynamic_import_tester
from paddlespeech.s2t.io.reader import LoadInputsAndTargets
from paddlespeech.s2t.models.asr_interface import ASRInterface
from paddlespeech.s2t.models.lm_interface import dynamic_import_lm
from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()

# NOTE: you need this func to generate our sphinx doc


def get_config(config_path):
    confs = CfgNode(new_allowed=True)
    confs.merge_from_file(config_path)
    return confs


def load_trained_model(args):
    confs = get_config(args.model_conf)
    class_obj = dynamic_import_tester(args.model_name)
    exp = class_obj(confs, args)
    with exp.eval():
        exp.setup()
        exp.restore()
    char_list = exp.args.char_list
    model = exp.model
    return model, char_list, exp, confs


def load_trained_lm(args):
    lm_args = get_config(args.rnnlm_conf)
    lm_model_module = lm_args.model_module
    lm_class = dynamic_import_lm(lm_model_module)
    lm = lm_class(**lm_args.model)
    model_dict = paddle.load(args.rnnlm)
    lm.set_state_dict(model_dict)
    return lm


def recog_v2(args):
    """Decode with custom models that implements ScorerInterface.

    Args:
        args (namespace): The program arguments.
        See py:func:`bin.asr_recog.get_parser` for details

    """
    logger.warning("experimental API for custom LMs is selected by --api v2")
    if args.batchsize > 1:
        raise NotImplementedError("multi-utt batch decoding is not implemented")
    if args.streaming_mode is not None:
        raise NotImplementedError("streaming mode is not implemented")
    if args.word_rnnlm:
        raise NotImplementedError("word LM is not implemented")

    # set_deterministic(args)
    model, char_list, exp, confs = load_trained_model(args)
    assert isinstance(model, ASRInterface)

    load_inputs_and_targets = LoadInputsAndTargets(
        mode="asr",
        load_output=False,
        sort_in_input_length=False,
        preprocess_conf=confs.preprocess_config
        if args.preprocess_conf is None else args.preprocess_conf,
        preprocess_args={"train": False}, )

    if args.rnnlm:
        lm = load_trained_lm(args)
        lm.eval()
    else:
        lm = None

    if args.ngram_model:
        from .scorers.ngram import NgramFullScorer
        from .scorers.ngram import NgramPartScorer

        if args.ngram_scorer == "full":
            ngram = NgramFullScorer(args.ngram_model, char_list)
        else:
            ngram = NgramPartScorer(args.ngram_model, char_list)
    else:
        ngram = None

    scorers = model.scorers()  # decoder
    scorers["lm"] = lm
    scorers["ngram"] = ngram
    scorers["length_bonus"] = LengthBonus(len(char_list))
    weights = dict(
        decoder=1.0 - args.ctc_weight,
        ctc=args.ctc_weight,
        lm=args.lm_weight,
        ngram=args.ngram_weight,
        length_bonus=args.penalty, )
    beam_search = BeamSearch(
        beam_size=args.beam_size,
        vocab_size=len(char_list),
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        eos=model.eos,
        token_list=char_list,
        pre_beam_score_key=None if args.ctc_weight == 1.0 else "full", )

    # TODO(karita): make all scorers batchfied
    if args.batchsize == 1:
        non_batch = [
            k for k, v in beam_search.full_scorers.items()
            if not isinstance(v, BatchScorerInterface)
        ]
        if len(non_batch) == 0:
            beam_search.__class__ = BatchBeamSearch
            logger.info("BatchBeamSearch implementation is selected.")
        else:
            logger.warning(f"As non-batch scorers {non_batch} are found, "
                           f"fall back to non-batch implementation.")

    if args.ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
    if args.ngpu == 1:
        device = "gpu:0"
    else:
        device = "cpu"
    paddle.set_device(device)
    dtype = getattr(paddle, args.dtype)
    logger.info(f"Decoding device={device}, dtype={dtype}")
    model.to(device=device, dtype=dtype)
    model.eval()
    beam_search.to(device=device, dtype=dtype)
    beam_search.eval()

    # read json data
    js = []
    with jsonlines.open(args.recog_json, "r") as reader:
        for item in reader:
            js.append(item)
    # jsonlines to dict, key by 'utt', value by jsonline
    js = {item['utt']: item for item in js}

    new_js = {}
    with paddle.no_grad():
        with jsonlines.open(args.result_label, "w") as f:
            for idx, name in enumerate(js.keys(), 1):
                logger.info(f"({idx}/{len(js.keys())}) decoding " + name)
                batch = [(name, js[name])]
                feat = load_inputs_and_targets(batch)[0][0]
                logger.info(f'feat: {feat.shape}')
                enc = model.encode(paddle.to_tensor(feat).to(dtype))
                logger.info(f'eout: {enc.shape}')
                nbest_hyps = beam_search(
                    x=enc,
                    maxlenratio=args.maxlenratio,
                    minlenratio=args.minlenratio)
                nbest_hyps = [
                    h.asdict()
                    for h in nbest_hyps[:min(len(nbest_hyps), args.nbest)]
                ]
                new_js[name] = add_results_to_json(js[name], nbest_hyps,
                                                   char_list)

                item = new_js[name]['output'][0]  # 1-best
                ref = item['text']
                rec_text = item['rec_text'].replace('▁', ' ').replace(
                    '<eos>', '').strip()
                rec_tokenid = list(map(int, item['rec_tokenid'].split()))
                f.write({
                    "utt": name,
                    "refs": [ref],
                    "hyps": [rec_text],
                    "hyps_tokenid": [rec_tokenid],
                })
