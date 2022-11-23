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
import argparse
import os
import re
from pathlib import Path
from typing import Dict
from typing import List
from typing import Union

DICT_EN = 'tools/aligner/cmudict-0.7b'
DICT_ZH = 'tools/aligner/simple.lexicon'
MODEL_DIR_EN = 'tools/aligner/vctk_model.zip'
MODEL_DIR_ZH = 'tools/aligner/aishell3_model.zip'
MFA_PHONE_EN = 'tools/aligner/vctk_model/meta.yaml'
MFA_PHONE_ZH = 'tools/aligner/aishell3_model/meta.yaml'
MFA_PATH = 'tools/montreal-forced-aligner/bin'
os.environ['PATH'] = MFA_PATH + '/:' + os.environ['PATH']


def check_phone(label_file: Union[str, Path],
                pronunciation_phones: Dict[str, str],
                mfa_phones: List[str],
                am_phones: List[str],
                oov_record: str="./oov_info.txt",
                lang: str="zh"):
    """Check whether the phoneme corresponding to the audio text content 
    is in the phoneme list of the pretrained mfa model to ensure that the alignment is normal.
    Check whether the phoneme corresponding to the audio text content 
    is in the phoneme list of the pretrained am model to ensure finetune (normalize) is normal.

    Args:
        label_file (Union[str, Path]): label file, format: utt_id|phone seq
        pronunciation_phones (dict): pronunciation to phones map dict
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
            transcription = re.sub(
                r'[：、，；。？！,.:;"?!”’《》【】<=>{}()（）#&@“”^_|…\\]', '',
                transcription)
            if lang == "en":
                transcription = transcription.upper()
            flag = 0
            temp_oov_words = []
            for word in transcription.split(" "):
                if word not in pronunciation_phones.keys():
                    temp_oov_words.append(word)
                    flag = 1
                    if word not in oov_words:
                        oov_words.append(word)
                else:
                    for p in pronunciation_phones[word]:
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


def get_pronunciation_phones(lexicon_file: Union[str, Path]):
    # pronunciation to phones
    pronunciation_phones = {}
    with open(lexicon_file, "r") as f2:
        for line in f2.readlines():
            line_list = line.strip().split(" ")
            pronunciation = line_list[0]
            if line_list[1] == '':
                phones = line_list[2:]
            else:
                phones = line_list[1:]
            pronunciation_phones[pronunciation] = phones

    return pronunciation_phones


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
                     am_phone_file: Union[str, Path],
                     input_dir: Union[str, Path],
                     newdir_name: str="newdir",
                     lang: str="zh"):
    """Check if there is any audio in the input that contains the oov word according to label_file.
       Copy audio that does not contain oov word to input_dir / newdir_name.
       Generate label file and save to input_dir / newdir_name.


    Args:
        label_file (Union[str, Path]): input audio label file, format: utt|pronunciation 
        am_phone_file (Union[str, Path]): pretrained am model phone file
        input_dir (Union[str, Path]): input dir
        newdir_name (str): directory name saved after checking oov
        lang (str): input audio language
    """

    if lang == 'en':
        lexicon_file = DICT_EN
        mfa_phone_file = MFA_PHONE_EN
    elif lang == 'zh':
        lexicon_file = DICT_ZH
        mfa_phone_file = MFA_PHONE_ZH
    else:
        print('please input right lang!!')

    pronunciation_phones = get_pronunciation_phones(lexicon_file)
    mfa_phones = get_mfa_phone(mfa_phone_file)
    am_phones = get_am_phone(am_phone_file)
    oov_words, oov_files, oov_file_words = check_phone(
        label_file=label_file,
        pronunciation_phones=pronunciation_phones,
        mfa_phones=mfa_phones,
        am_phones=am_phones,
        oov_record="./oov_info.txt",
        lang=lang)

    input_dir = Path(input_dir).expanduser()
    new_dir = input_dir / newdir_name
    new_dir.mkdir(parents=True, exist_ok=True)
    with open(label_file, "r") as f:
        for line in f.readlines():
            utt_id = line.split("|")[0]
            if utt_id not in oov_files:
                transcription = line.split("|")[1].strip()
                wav_file = str(input_dir) + "/" + utt_id + ".wav"
                new_wav_file = str(new_dir) + "/" + utt_id + ".wav"
                os.system("cp %s %s" % (wav_file, new_wav_file))
                single_file = str(new_dir) + "/" + utt_id + ".txt"
                with open(single_file, "w") as fw:
                    fw.write(transcription)


if __name__ == '__main__':
    # parse config and args
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features.")

    parser.add_argument(
        "--input_dir",
        type=str,
        default="./input/csmsc_mini",
        help="directory containing audio and label file")

    parser.add_argument(
        "--pretrained_model_dir",
        type=str,
        default="./pretrained_models/fastspeech2_aishell3_ckpt_1.1.0",
        help="Path to pretrained model")

    parser.add_argument(
        "--newdir_name",
        type=str,
        default="newdir",
        help="directory name saved after checking oov")

    parser.add_argument(
        '--lang',
        type=str,
        default='zh',
        choices=['zh', 'en'],
        help='Choose input audio language. zh or en')

    args = parser.parse_args()

    # if args.lang == 'en':
    #     lexicon_file = DICT_EN
    #     mfa_phone_file = MFA_PHONE_EN
    # elif args.lang == 'zh':
    #     lexicon_file = DICT_ZH
    #     mfa_phone_file = MFA_PHONE_ZH
    # else:
    #     print('please input right lang!!')
    assert args.lang == "zh" or args.lang == "en", "please input right lang! zh or en"

    input_dir = Path(args.input_dir).expanduser()
    pretrained_model_dir = Path(args.pretrained_model_dir).expanduser()
    am_phone_file = pretrained_model_dir / "phone_id_map.txt"
    label_file = input_dir / "labels.txt"

    get_check_result(
        label_file=label_file,
        am_phone_file=am_phone_file,
        input_dir=input_dir,
        newdir_name=args.newdir_name,
        lang=args.lang)
