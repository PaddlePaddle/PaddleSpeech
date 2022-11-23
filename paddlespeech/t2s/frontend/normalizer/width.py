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


def full2half_width(ustr):
    half = []
    for u in ustr:
        num = ord(u)
        if num == 0x3000:  # 全角空格变半角
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        u = chr(num)
        half.append(u)
    return ''.join(half)


def half2full_width(ustr):
    full = []
    for u in ustr:
        num = ord(u)
        if num == 32:  # 半角空格变全角
            num = 0x3000
        elif 0x21 <= num <= 0x7E:
            num += 0xfee0
        u = chr(num)  # to unicode
        full.append(u)

    return ''.join(full)
