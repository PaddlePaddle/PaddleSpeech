"""V2 backend for `asr_recog.py` using py:class:`espnet.nets.beam_search.BeamSearch`."""

import json
import paddle
import yaml
from yacs.config import CfgNode
from pathlib import Path
import jsonlines

# from espnet.asr.asr_utils import get_model_conf
# from espnet.asr.asr_utils import torch_load
# from espnet.asr.pytorch_backend.asr import load_trained_model
# from espnet.nets.lm_interface import dynamic_import_lm

from deepspeech.models.asr_interface import ASRInterface

from .utils import add_results_to_json
# from .batch_beam_search import BatchBeamSearch
from .beam_search import BeamSearch
from .scorers.scorer_interface import BatchScorerInterface
from .scorers.length_bonus import LengthBonus

from deepspeech.io.reader import LoadInputsAndTargets
from deepspeech.utils.log import Log
logger = Log(__name__).getlog()


from deepspeech.utils.dynamic_import import dynamic_import
from deepspeech.utils.utility import print_arguments

model_test_alias = {
    "u2": "deepspeech.exps.u2.model:U2Tester",
    "u2_kaldi": "deepspeech.exps.u2_kaldi.model:U2Tester",
}

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
    args.nprocs = args.ngpu
    # set_deterministic(args)

    #model, train_args = load_trained_model(args.model)
    model_path = Path(args.model)
    ckpt_dir = model_path.parent.parent

    confs = CfgNode()
    confs.set_new_allowed(True)
    confs.merge_from_file(args.model_conf)

    class_obj = dynamic_import(args.model_name, model_test_alias)
    exp = class_obj(confs, args)
    with exp.eval():
        exp.setup()
        exp.restore()
    char_list = exp.args.char_list

    model = exp.model
    assert isinstance(model, ASRInterface)
    load_inputs_and_targets = LoadInputsAndTargets(
        mode="asr",
        load_output=False,
        sort_in_input_length=False,
        preprocess_conf=confs.collator.augmentation_config
        if args.preprocess_conf is None
        else args.preprocess_conf,
        preprocess_args={"train": False},
    )

    if args.rnnlm:
        lm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        # NOTE: for a compatibility with less than 0.5.0 version models
        lm_model_module = getattr(lm_args, "model_module", "default")
        lm_class = dynamic_import_lm(lm_model_module, lm_args.backend)
        lm = lm_class(len(char_list), lm_args)
        torch_load(args.rnnlm, lm)
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

    scorers = model.scorers()
    scorers["lm"] = lm
    scorers["ngram"] = ngram
    scorers["length_bonus"] = LengthBonus(len(char_list))
    weights = dict(
        decoder=1.0 - args.ctc_weight,
        ctc=args.ctc_weight,
        lm=args.lm_weight,
        ngram=args.ngram_weight,
        length_bonus=args.penalty,
    )
    beam_search = BeamSearch(
        beam_size=args.beam_size,
        vocab_size=len(char_list),
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        eos=model.eos,
        token_list=char_list,
        pre_beam_score_key=None if args.ctc_weight == 1.0 else "full",
    )

    # TODO(karita): make all scorers batchfied
    if args.batchsize == 1:
        non_batch = [
            k
            for k, v in beam_search.full_scorers.items()
            if not isinstance(v, BatchScorerInterface)
        ]
        if len(non_batch) == 0:
            beam_search.__class__ = BatchBeamSearch
            logger.info("BatchBeamSearch implementation is selected.")
        else:
            logger.warning(
                f"As non-batch scorers {non_batch} are found, "
                f"fall back to non-batch implementation."
            )

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
    # josnlines to dict, key by 'utt'
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
                logger.info(f'eouts: {enc.shape}')
                nbest_hyps = beam_search(
                    x=enc, maxlenratio=args.maxlenratio, minlenratio=args.minlenratio
                )
                nbest_hyps = [
                    h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), args.nbest)]
                ]
                new_js[name] = add_results_to_json(
                    js[name], nbest_hyps, char_list
                )

                item = new_js[name]['output'][0] # 1-best
                utt = name 
                ref = item['text']
                rec_text = item['rec_text'].replace('▁', ' ').strip()
                rec_tokenid = item['rec_tokenid'].split()
                f.write({
                        "utt": utt,
                        "refs": [ref],
                        "hyps": [rec_text],
                        "hyps_tokenid": [rec_tokenid],
                    })