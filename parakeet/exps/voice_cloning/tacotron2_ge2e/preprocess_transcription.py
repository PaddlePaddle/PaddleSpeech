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
import pickle
import re
from pathlib import Path

import tqdm
import yaml

zh_pattern = re.compile("[\u4e00-\u9fa5]")

_tones = {'<pad>', '<s>', '</s>', '0', '1', '2', '3', '4', '5'}

_pauses = {'%', '$'}

_initials = {
    'b',
    'p',
    'm',
    'f',
    'd',
    't',
    'n',
    'l',
    'g',
    'k',
    'h',
    'j',
    'q',
    'x',
    'zh',
    'ch',
    'sh',
    'r',
    'z',
    'c',
    's',
}

_finals = {
    'ii',
    'iii',
    'a',
    'o',
    'e',
    'ea',
    'ai',
    'ei',
    'ao',
    'ou',
    'an',
    'en',
    'ang',
    'eng',
    'er',
    'i',
    'ia',
    'io',
    'ie',
    'iai',
    'iao',
    'iou',
    'ian',
    'ien',
    'iang',
    'ieng',
    'u',
    'ua',
    'uo',
    'uai',
    'uei',
    'uan',
    'uen',
    'uang',
    'ueng',
    'v',
    've',
    'van',
    'ven',
    'veng',
}

_ernized_symbol = {'&r'}

_specials = {'<pad>', '<unk>', '<s>', '</s>'}

_phones = _initials | _finals | _ernized_symbol | _specials | _pauses


def is_zh(word):
    global zh_pattern
    match = zh_pattern.search(word)
    return match is not None


def ernized(syllable):
    return syllable[:2] != "er" and syllable[-2] == 'r'


def convert(syllable):
    # expansion of o -> uo
    syllable = re.sub(r"([bpmf])o$", r"\1uo", syllable)
    # syllable = syllable.replace("bo", "buo").replace("po", "puo").replace("mo", "muo").replace("fo", "fuo")
    # expansion for iong, ong
    syllable = syllable.replace("iong", "veng").replace("ong", "ueng")

    # expansion for ing, in
    syllable = syllable.replace("ing", "ieng").replace("in", "ien")

    # expansion for un, ui, iu
    syllable = syllable.replace("un", "uen").replace("ui",
                                                     "uei").replace("iu", "iou")

    # rule for variants of i
    syllable = syllable.replace("zi", "zii").replace("ci", "cii").replace("si", "sii")\
        .replace("zhi", "zhiii").replace("chi", "chiii").replace("shi", "shiii")\
        .replace("ri", "riii")

    # rule for y preceding i, u
    syllable = syllable.replace("yi", "i").replace("yu", "v").replace("y", "i")

    # rule for w
    syllable = syllable.replace("wu", "u").replace("w", "u")

    # rule for v following j, q, x
    syllable = syllable.replace("ju", "jv").replace("qu",
                                                    "qv").replace("xu", "xv")

    return syllable


def split_syllable(syllable: str):
    """Split a syllable in pinyin into a list of phones and a list of tones.
    Initials have no tone, represented by '0', while finals have tones from
    '1,2,3,4,5'.

    e.g.

    zhang -> ['zh', 'ang'], ['0', '1']
    """
    if syllable in _pauses:
        # syllable, tone
        return [syllable], ['0']

    tone = syllable[-1]
    syllable = convert(syllable[:-1])

    phones = []
    tones = []

    global _initials
    if syllable[:2] in _initials:
        phones.append(syllable[:2])
        tones.append('0')
        phones.append(syllable[2:])
        tones.append(tone)
    elif syllable[0] in _initials:
        phones.append(syllable[0])
        tones.append('0')
        phones.append(syllable[1:])
        tones.append(tone)
    else:
        phones.append(syllable)
        tones.append(tone)
    return phones, tones


def load_aishell3_transcription(line: str):
    sentence_id, pinyin, text = line.strip().split("|")
    syllables = pinyin.strip().split()

    results = []

    for syllable in syllables:
        if syllable in _pauses:
            results.append(syllable)
        elif not ernized(syllable):
            results.append(syllable)
        else:
            results.append(syllable[:-2] + syllable[-1])
            results.append('&r5')

    phones = []
    tones = []
    for syllable in results:
        p, t = split_syllable(syllable)
        phones.extend(p)
        tones.extend(t)
    for p in phones:
        assert p in _phones, p
    return {
        "sentence_id": sentence_id,
        "text": text,
        "syllables": results,
        "phones": phones,
        "tones": tones
    }


def process_aishell3(dataset_root, output_dir):
    dataset_root = Path(dataset_root).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    prosody_label_path = dataset_root / "label_train-set.txt"
    with open(prosody_label_path, 'rt') as f:
        lines = [line.strip() for line in f]

    records = lines[5:]

    processed_records = []
    for record in tqdm.tqdm(records):
        new_record = load_aishell3_transcription(record)
        processed_records.append(new_record)
        print(new_record)

    with open(output_dir / "metadata.pickle", 'wb') as f:
        pickle.dump(processed_records, f)

    with open(output_dir / "metadata.yaml", 'wt', encoding="utf-8") as f:
        yaml.safe_dump(
            processed_records, f, default_flow_style=None, allow_unicode=True)

    print("metadata done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess transcription of AiShell3 and save them in a compact file(yaml and pickle)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="~/datasets/aishell3/train",
        help="path of the training dataset,(contains a label_train-set.txt).")
    parser.add_argument(
        "--output",
        type=str,
        help="the directory to save the processed transcription."
        "If not provided, it would be the same as the input.")
    args = parser.parse_args()
    if args.output is None:
        args.output = args.input

    process_aishell3(args.input, args.output)
