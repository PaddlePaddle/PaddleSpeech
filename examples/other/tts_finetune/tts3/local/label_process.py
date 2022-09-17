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
import os
from pathlib import Path
from typing import List
from typing import Union


def change_baker_label(baker_label_file: Union[str, Path],
                       out_label_file: Union[str, Path]):
    """change baker label file to regular label file

    Args:
        baker_label_file (Union[str, Path]): Original baker label file
        out_label_file (Union[str, Path]): regular label file
    """
    with open(baker_label_file) as f:
        lines = f.readlines()

    with open(out_label_file, "w") as fw:
        for i in range(0, len(lines), 2):
            utt_id = lines[i].split()[0]
            transcription = lines[i + 1].strip()
            fw.write(utt_id + "|" + transcription + "\n")


def get_single_label(label_file: Union[str, Path],
                     oov_files: List[Union[str, Path]],
                     input_dir: Union[str, Path]):
    """Divide the label file into individual files according to label_file

    Args:
        label_file (str or Path): label file, format: utt_id|phones id
        input_dir (Path): input dir including audios
    """
    input_dir = Path(input_dir).expanduser()
    new_dir = input_dir / "newdir"
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

    return new_dir
