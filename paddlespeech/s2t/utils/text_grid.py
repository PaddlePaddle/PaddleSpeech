# Copyright 2021 Mobvoi Inc. All Rights Reserved.
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
# Modified from wenet(https://github.com/wenet-e2e/wenet)
from typing import Dict
from typing import List
from typing import Text

import textgrid


def segment_alignment(alignment: List[int], blank_id=0) -> List[List[int]]:
    """segment ctc alignment ids by continuous blank and repeat label.

    Args:
        alignment (List[int]): ctc alignment id sequence. 
            e.g. [0, 0, 0, 1, 1, 1, 2, 0, 0, 3]
        blank_id (int, optional): blank id. Defaults to 0.

    Returns:
        List[List[int]]: token align, segment aligment id sequence. 
            e.g. [[0, 0, 0, 1, 1, 1], [2], [0, 0, 3]]
    """
    # convert alignment to a praat format, which is a doing phonetics
    # by computer and helps analyzing alignment
    align_segs = []
    # get frames level duration for each token
    start = 0
    end = 0
    while end < len(alignment):
        while end < len(alignment) and alignment[end] == blank_id:  # blank
            end += 1
        if end == len(alignment):
            align_segs[-1].extend(alignment[start:])
            break
        end += 1
        while end < len(alignment) and alignment[end - 1] == alignment[
                end]:  # repeat label
            end += 1
        align_segs.append(alignment[start:end])
        start = end
    return align_segs


def align_to_tierformat(align_segs: List[List[int]],
                        subsample: int,
                        token_dict: Dict[int, Text],
                        blank_id=0) -> List[Text]:
    """Generate textgrid.Interval format from alignment segmentations.

    Args:
        align_segs (List[List[int]]): segmented ctc alignment ids.
        subsample (int): 25ms frame_length, 10ms hop_length, 1/subsample
        token_dict (Dict[int, Text]): int -> str map.

    Returns:
        List[Text]: list of textgrid.Interval text, str(start, end, text).
    """
    hop_length = 10  # ms
    second_ms = 1000  # ms
    frame_per_second = second_ms / hop_length  # 25ms frame_length, 10ms hop_length
    second_per_frame = 1.0 / frame_per_second

    begin = 0
    duration = 0
    tierformat = []

    for idx, tokens in enumerate(align_segs):
        token_len = len(tokens)
        token = tokens[-1]
        # time duration in second
        duration = token_len * subsample * second_per_frame
        if idx < len(align_segs) - 1:
            print(f"{begin:.2f} {begin + duration:.2f} {token_dict[token]}")
            tierformat.append(
                f"{begin:.2f} {begin + duration:.2f} {token_dict[token]}\n")
        else:
            for i in tokens:
                if i != blank_id:
                    token = i
                    break
            print(f"{begin:.2f} {begin + duration:.2f} {token_dict[token]}")
            tierformat.append(
                f"{begin:.2f} {begin + duration:.2f} {token_dict[token]}\n")
        begin = begin + duration

    return tierformat


def generate_textgrid(maxtime: float,
                      intervals: List[Text],
                      output: Text,
                      name: Text='ali') -> None:
    """Create alignment textgrid file.

    Args:
        maxtime (float): audio duartion.
        intervals (List[Text]): ctc output alignment. e.g. "start-time end-time word" per item.
        output (Text): textgrid filepath.
        name (Text, optional): tier or layer name. Defaults to 'ali'.
    """
    # Download Praat: https://www.fon.hum.uva.nl/praat/
    avg_interval = maxtime / (len(intervals) + 1)
    print(f"average second/token: {avg_interval}")
    margin = 0.0001

    tg = textgrid.TextGrid(maxTime=maxtime)
    tier = textgrid.IntervalTier(name=name, maxTime=maxtime)

    i = 0
    for dur in intervals:
        s, e, text = dur.split()
        tier.add(minTime=float(s) + margin, maxTime=float(e), mark=text)

    tg.append(tier)

    tg.write(output)
    print("successfully generator textgrid {}.".format(output))
