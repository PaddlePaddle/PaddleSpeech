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
# Modified from espnet(https://github.com/espnet/espnet)
import numpy as np

from paddlespeech.s2t.utils.log import Log
logger = Log(__name__).getlog()

__all__ = ["end_detect", "parse_hypothesis", "add_results_to_json"]


def end_detect(ended_hyps, i, M=3, D_end=np.log(1 * np.exp(-10))):
    """End detection.

    described in Eq. (50) of S. Watanabe et al
    "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition"

    :param ended_hyps: dict
    :param i: int
    :param M: int
    :param D_end: float
    :return: bool
    """
    if len(ended_hyps) == 0:
        return False
    count = 0
    best_hyp = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[0]
    for m in range(M):
        # get ended_hyps with their length is i - m
        hyp_length = i - m
        hyps_same_length = [
            x for x in ended_hyps if len(x["yseq"]) == hyp_length
        ]
        if len(hyps_same_length) > 0:
            best_hyp_same_length = sorted(
                hyps_same_length, key=lambda x: x["score"], reverse=True)[0]
            if best_hyp_same_length["score"] - best_hyp["score"] < D_end:
                count += 1

    if count == M:
        return True
    else:
        return False


# * ------------------ recognition related ------------------ *
def parse_hypothesis(hyp, char_list):
    """Parse hypothesis.

    Args:
        hyp (list[dict[str, Any]]): Recognition hypothesis.
        char_list (list[str]): List of characters.

    Returns:
        tuple(str, str, str, float)

    """
    # remove sos and get results
    tokenid_as_list = list(map(int, hyp["yseq"][1:]))
    token_as_list = [char_list[idx] for idx in tokenid_as_list]
    score = float(hyp["score"])

    # convert to string
    tokenid = " ".join([str(idx) for idx in tokenid_as_list])
    token = " ".join(token_as_list)
    text = "".join(token_as_list).replace("<space>", " ")

    return text, token, tokenid, score


def add_results_to_json(js, nbest_hyps, char_list):
    """Add N-best results to json.

    Args:
        js (dict[str, Any]): Groundtruth utterance dict.
        nbest_hyps_sd (list[dict[str, Any]]):
            List of hypothesis for multi_speakers: nutts x nspkrs.
        char_list (list[str]): List of characters.

    Returns:
        dict[str, Any]: N-best results added utterance dict.

    """
    # copy old json info
    new_js = dict()
    new_js["utt2spk"] = js["utt2spk"]
    new_js["output"] = []

    for n, hyp in enumerate(nbest_hyps, 1):
        # parse hypothesis
        rec_text, rec_token, rec_tokenid, score = parse_hypothesis(hyp,
                                                                   char_list)

        # copy ground-truth
        if len(js["output"]) > 0:
            out_dic = dict(js["output"][0].items())
        else:
            # for no reference case (e.g., speech translation)
            out_dic = {"name": ""}

        # update name
        out_dic["name"] += "[%d]" % n

        # add recognition results
        out_dic["rec_text"] = rec_text
        out_dic["rec_token"] = rec_token
        out_dic["rec_tokenid"] = rec_tokenid
        out_dic["score"] = score

        # add to list of N-best result dicts
        new_js["output"].append(out_dic)

        # show 1-best result
        if n == 1:
            if "text" in out_dic.keys():
                logger.info("groundtruth: %s" % out_dic["text"])
            logger.info("prediction : %s" % out_dic["rec_text"])

    return new_js
