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
from pathlib import Path
from typing import Union

DICT_EN = 'tools/aligner/cmudict-0.7b'
DICT_ZH = 'tools/aligner/simple.lexicon'
MODEL_DIR_EN = 'tools/aligner/vctk_model.zip'
MODEL_DIR_ZH = 'tools/aligner/aishell3_model.zip'
MFA_PHONE_EN = 'tools/aligner/vctk_model/meta.yaml'
MFA_PHONE_ZH = 'tools/aligner/aishell3_model/meta.yaml'
MFA_PATH = 'tools/montreal-forced-aligner/bin'
os.environ['PATH'] = MFA_PATH + '/:' + os.environ['PATH']


def get_mfa_result(
        input_dir: Union[str, Path],
        mfa_dir: Union[str, Path],
        lang: str='en', ):
    """get mfa result

    Args:
        input_dir (Union[str, Path]): input dir including wav file and label
        mfa_dir (Union[str, Path]): mfa result dir
        lang (str, optional): input audio language. Defaults to 'en'.
    """
    # MFA
    if lang == 'en':
        DICT = DICT_EN
        MODEL_DIR = MODEL_DIR_EN

    elif lang == 'zh':
        DICT = DICT_ZH
        MODEL_DIR = MODEL_DIR_ZH
    else:
        print('please input right lang!!')

    CMD = 'mfa_align' + ' ' + str(
        input_dir) + ' ' + DICT + ' ' + MODEL_DIR + ' ' + str(mfa_dir)
    os.system(CMD)


if __name__ == '__main__':
    # parse config and args
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features.")

    parser.add_argument(
        "--input_dir",
        type=str,
        default="./input/baker_mini/newdir",
        help="directory containing audio and label file")

    parser.add_argument(
        "--mfa_dir",
        type=str,
        default="./mfa_result",
        help="directory to save aligned files")

    parser.add_argument(
        '--lang',
        type=str,
        default='zh',
        choices=['zh', 'en'],
        help='Choose input audio language. zh or en')

    args = parser.parse_args()

    get_mfa_result(
        input_dir=args.input_dir, mfa_dir=args.mfa_dir, lang=args.lang)
