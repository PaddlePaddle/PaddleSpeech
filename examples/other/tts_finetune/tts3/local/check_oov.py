# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from pathlib import Path
from typing import Dict
from typing import List
from typing import Union


def check_phone(label_file: Union[str, Path],
                pinyin_phones: Dict[str, str],
                mfa_phones: List[str],
                am_phones: List[str],
                oov_record: str="./oov_info.txt"):
    """Check whether the phoneme corresponding to the audio text content 
    is in the phoneme list of the pretrained mfa model to ensure that the alignment is normal.
    Check whether the phoneme corresponding to the audio text content 
    is in the phoneme list of the pretrained am model to ensure finetune (normalize) is normal.

    Args:
        label_file (Union[str, Path]): label file, format: utt_id|phone seq
        pinyin_phones (dict): pinyin to phones map dict
        mfa_phones (list): the phone list of pretrained mfa model
        am_phones (list): the phone list of pretrained mfa model

    Returns:
        oov_words (list): oov words
        oov_files (list): utt id list that exist oov
        oov_file_words (dict): the oov file and oov phone in this file
    """
    oov_words = []
    oov_files = []
    oov_file_words = {}

    with open(label_file, "r") as f:
        for line in f.readlines():
            utt_id = line.split("|")[0]
            transcription = line.strip().split("|")[1]
            flag = 0
            temp_oov_words = []
            for word in transcription.split(" "):
                if word not in pinyin_phones.keys():
                    temp_oov_words.append(word)
                    flag = 1
                    if word not in oov_words:
                        oov_words.append(word)
                else:
                    for p in pinyin_phones[word]:
                        if p not in mfa_phones or p not in am_phones:
                            temp_oov_words.append(word)
                            flag = 1
                            if word not in oov_words:
                                oov_words.append(word)
            if flag == 1:
                oov_files.append(utt_id)
                oov_file_words[utt_id] = temp_oov_words

    if oov_record is not None:
        with open(oov_record, "w") as fw:
            fw.write("oov_words: " + str(oov_words) + "\n")
            fw.write("oov_files: " + str(oov_files) + "\n")
            fw.write("oov_file_words: " + str(oov_file_words) + "\n")

    return oov_words, oov_files, oov_file_words


def get_pinyin_phones(lexicon_file: Union[str, Path]):
    # pinyin to phones
    pinyin_phones = {}
    with open(lexicon_file, "r") as f2:
        for line in f2.readlines():
            line_list = line.strip().split(" ")
            pinyin = line_list[0]
            if line_list[1] == '':
                phones = line_list[2:]
            else:
                phones = line_list[1:]
            pinyin_phones[pinyin] = phones

    return pinyin_phones


def get_mfa_phone(mfa_phone_file: Union[str, Path]):
    # get phones from pretrained mfa model (meta.yaml)
    mfa_phones = []
    with open(mfa_phone_file, "r") as f:
        for line in f.readlines():
            if line.startswith("-"):
                phone = line.strip().split(" ")[-1]
                mfa_phones.append(phone)

    return mfa_phones


def get_am_phone(am_phone_file: Union[str, Path]):
    # get phones from pretrained am model (phone_id_map.txt)
    am_phones = []
    with open(am_phone_file, "r") as f:
        for line in f.readlines():
            phone = line.strip().split(" ")[0]
            am_phones.append(phone)

    return am_phones


def get_check_result(label_file: Union[str, Path],
                     lexicon_file: Union[str, Path],
                     mfa_phone_file: Union[str, Path],
                     am_phone_file: Union[str, Path]):
    pinyin_phones = get_pinyin_phones(lexicon_file)
    mfa_phones = get_mfa_phone(mfa_phone_file)
    am_phones = get_am_phone(am_phone_file)
    oov_words, oov_files, oov_file_words = check_phone(
        label_file, pinyin_phones, mfa_phones, am_phones)
    return oov_words, oov_files, oov_file_words
