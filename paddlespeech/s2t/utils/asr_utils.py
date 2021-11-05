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
import json

import numpy as np

__all__ = ["label_smoothing_dist"]


def label_smoothing_dist(odim, lsm_type, transcript=None, blank=0):
    """Obtain label distribution for loss smoothing.

    :param odim:
    :param lsm_type:
    :param blank:
    :param transcript:
    :return:
    """
    if transcript is not None:
        with open(transcript, "rb") as f:
            trans_json = json.load(f)["utts"]

    if lsm_type == "unigram":
        assert transcript is not None, (
            "transcript is required for %s label smoothing" % lsm_type)
        labelcount = np.zeros(odim)
        for k, v in trans_json.items():
            ids = np.array([int(n) for n in v["output"][0]["tokenid"].split()])
            # to avoid an error when there is no text in an uttrance
            if len(ids) > 0:
                labelcount[ids] += 1
        labelcount[odim - 1] = len(transcript)  # count <eos>
        labelcount[labelcount == 0] = 1  # flooring
        labelcount[blank] = 0  # remove counts for blank
        labeldist = labelcount.astype(np.float32) / np.sum(labelcount)
    else:
        logging.error("Error: unexpected label smoothing type: %s" % lsm_type)
        sys.exit()

    return labeldist
