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
import argparse
from typing import List, Union
from pathlib import Path


def erized(syllable: str) -> bool:
    """Whether the syllable contains erhua effect.
    
    Example
    --------
    huar -> True
    guanr -> True
    er -> False
    """
    # note: for pinyin, len(syllable) >=2 is always true
    # if not: there is something wrong in the data
    assert len(syllable) >= 2, f"inavlid syllable {syllable}"
    return syllable[:2] != "er" and syllable[-2] == 'r'


def ignore_sandhi(reference: List[str], generated: List[str]) -> List[str]:
    """
    Given a sequence of syllables from human annotation(reference), 
    which makes sandhi explici and a sequence of syllables from some 
    simple g2p program(generated), which does not consider sandhi, 
    return a the reference sequence while ignore sandhi.

    Example
    --------
    ['lao2', 'hu3'], ['lao3', 'hu3'] -> ['lao3', 'hu3']
    """
    i = 0
    j = 0

    # sandhi ignored in the result while other errors are not included
    result = []
    while i < len(reference):
        if erized(reference[i]):
            result.append(reference[i])
            i += 1
            j += 2
        elif reference[i][:-1] == generated[i][:-1] and reference[i][
                -1] == '2' and generated[i][-1] == '3':
            result.append(generated[i])
            i += 1
            j += 1
        else:
            result.append(reference[i])
            i += 1
            j += 1
    assert j == len(
        generated
    ), "length of transcriptions mismatch, There may be some characters that are ignored in the generated transcription."
    return result


def convert_transcriptions(reference: Union[str, Path], generated: Union[str, Path], output: Union[str, Path]):
    with open(reference, 'rt') as f_ref:
        with open(generated, 'rt') as f_gen:
            with open(output, 'wt') as f_out:
                for i, (ref, gen) in enumerate(zip(f_ref, f_gen)):
                    sentence_id, ref_transcription = ref.strip().split(' ', 1)
                    _, gen_transcription = gen.strip().split(' ', 1)
                    try:
                        result = ignore_sandhi(ref_transcription.split(),
                                               gen_transcription.split())
                        result = ' '.join(result)
                    except Exception:
                        print(
                            f"sentence_id: {sentence_id} There is some annotation error in the reference or generated transcription. Use the reference."
                        )
                        result = ref_transcription
                    f_out.write(f"{sentence_id} {result}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="reference transcription but ignore sandhi.")
    parser.add_argument(
        "--reference",
        type=str,
        help="path to the reference transcription of baker dataset.")
    parser.add_argument(
        "--generated", type=str, help="path to the generated transcription.")
    parser.add_argument("--output", type=str, help="path to save result.")
    args = parser.parse_args()
    convert_transcriptions(args.reference, args.generated, args.output)
