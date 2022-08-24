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


def generate_finetune_env(output_dir: Path, pretrained_model_dir: Path):

    output_dir = output_dir / "checkpoints/"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = sorted(list((pretrained_model_dir).rglob("*.pdz")))[0]
    model_path = model_path.resolve()
    iter = int(str(model_path).split("_")[-1].split(".")[0])
    model_file = str(model_path).split("/")[-1]

    os.system("cp %s %s" % (model_path, output_dir))

    records_file = output_dir / "records.jsonl"
    with open(records_file, "w") as f:
        line = "\"time\": \"2022-08-06 07:51:53.463650\", \"path\": \"%s\", \"iteration\": %d" % (
            str(output_dir / model_file), iter)
        f.write("{" + line + "}" + "\n")
