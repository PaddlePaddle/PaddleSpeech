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
import os
import re
from pathlib import Path

import paddle
from paddleocr import draw_ocr
from paddleocr import PaddleOCR
from PIL import Image


def evaluate(args, ocr):
    img_dir = Path(args.img_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    img_out_dir = output_dir / "imgs"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "sentences.txt", "w") as wf:
        for name in os.listdir(img_dir):
            id = name.split(".")[0]
            img_path = img_dir / name
            result = ocr.ocr(str(img_path), cls=True)
            # draw result
            image = Image.open(img_path).convert('RGB')
            boxes = [line[0] for line in result]
            txts = [line[1][0] for line in result]
            scores = [line[1][1] for line in result]
            im_show = draw_ocr(
                image, boxes, txts, scores, font_path=args.font_path)
            im_show = Image.fromarray(im_show)
            paragraph = "".join(txts)
            # 过滤出中文结果
            pattern = re.compile(r'[^(\u4e00-\u9fa5)+，。？、]')
            sentence = re.sub(pattern, '', paragraph)
            im_show.save(img_out_dir / name)
            wf.write(id + " " + sentence + "\n")


def main():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(
        description="Synthesize with fastspeech2 & parallel wavegan.")
    parser.add_argument("--img-dir", default="imgs", type=str, help="img_dir.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="output sentences path.")
    parser.add_argument(
        "--font-path", type=str, default="simfang.ttf", help="font path")
    args = parser.parse_args()

    paddle.set_device("gpu")
    # need to run only once to download and load model into memory
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')

    evaluate(args, ocr)


if __name__ == "__main__":
    main()
