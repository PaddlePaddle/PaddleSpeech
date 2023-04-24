# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

PHONESFILE = "./dict/phones.txt"
PHONES_ID_FILE = "./dict/phonesid.dict"
TONESFILE = "./dict/tones.txt"
TONES_ID_FILE = "./dict/tonesid.dict"


def GenIdFile(file, idfile):
    id = 2
    with open(file, 'r') as f1, open(idfile, "w+") as f2:
        f2.write("<pad> 0\n")
        f2.write("<unk> 1\n")
        for line in f1.readlines():
            phone = line.strip()
            print(phone + " " + str(id) + "\n")
            f2.write(phone + " " + str(id) + "\n")
            id += 1


if __name__ == "__main__":
    GenIdFile(PHONESFILE, PHONES_ID_FILE)
    GenIdFile(TONESFILE, TONES_ID_FILE)
