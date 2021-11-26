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
import logging
from collections import OrderedDict
from pathlib import Path


def detect_oov(corpus_dir, lexicon_path, transcription_pattern="*.lab"):
    corpus_dir = Path(corpus_dir)

    lexicon = OrderedDict()
    with open(lexicon_path, 'rt') as f:
        for line in f:
            syllable, phonemes = line.split(maxsplit=1)
            lexicon[syllable] = phonemes

    for fp in corpus_dir.glob(transcription_pattern):
        syllables = fp.read_text().strip().split()
        for s in syllables:
            if s not in lexicon:
                logging.warning(f"{fp.relative_to(corpus_dir)} has OOV {s} .")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="detect oov in a corpus given a lexicon")
    parser.add_argument(
        "corpus_dir", type=str, help="corpus dir for MFA alignment.")
    parser.add_argument("lexicon_path", type=str, help="dictionary to use.")
    parser.add_argument(
        "--pattern", type=str, default="*.lab", help="dictionary to use.")
    args = parser.parse_args()
    print(args)

    detect_oov(args.corpus_dir, args.lexicon_path, args.pattern)
