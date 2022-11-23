# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

__all__ = ["get_punctuations"]

EN_PUNCT = [
    " ",
    "-",
    "...",
    ",",
    ".",
    "?",
    "!",
]

CN_PUNCT = ["、", "，", "；", "：", "。", "？", "！"]


def get_punctuations(lang):
    if lang == "en":
        return EN_PUNCT
    elif lang == "cn":
        return CN_PUNCT
    else:
        raise ValueError(f"language {lang} Not supported")
