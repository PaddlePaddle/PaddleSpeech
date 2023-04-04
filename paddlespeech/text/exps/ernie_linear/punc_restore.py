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
import re

import paddle
import yaml
from paddlenlp.transformers import ErnieTokenizer
from yacs.config import CfgNode

from paddlespeech.text.models.ernie_linear import ErnieLinear

DefinedClassifier = {
    'ErnieLinear': ErnieLinear,
}


def _clean_text(text, punc_list):
    text = text.lower()
    text = re.sub('[^A-Za-z0-9\u4e00-\u9fa5]', '', text)
    text = re.sub(f'[{"".join([p for p in punc_list][1:])}]', '', text)
    return text


def preprocess(text, punc_list, tokenizer):
    clean_text = _clean_text(text, punc_list)
    assert len(clean_text) > 0, f'Invalid input string: {text}'
    tokenized_input = tokenizer(list(clean_text),
                                return_length=True,
                                is_split_into_words=True)
    _inputs = dict()
    _inputs['input_ids'] = tokenized_input['input_ids']
    _inputs['seg_ids'] = tokenized_input['token_type_ids']
    _inputs['seq_len'] = tokenized_input['seq_len']
    return _inputs


def test(args):
    with open(args.config) as f:
        config = CfgNode(yaml.safe_load(f))
    print("========Args========")
    print(yaml.safe_dump(vars(args), allow_unicode=True))
    # print(args)
    print("========Config========")
    print(config)

    punc_list = []
    with open(config["data_params"]["punc_path"], 'r') as f:
        for line in f:
            punc_list.append(line.strip())

    model = DefinedClassifier[config["model_type"]](**config["model"])
    # print(model)

    pretrained_token = config['data_params']['pretrained_token']
    tokenizer = ErnieTokenizer.from_pretrained(pretrained_token)
    # tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')

    state_dict = paddle.load(args.checkpoint)
    model.set_state_dict(state_dict["main_params"])
    model.eval()
    _inputs = preprocess(args.text, punc_list, tokenizer)
    seq_len = _inputs['seq_len']
    input_ids = paddle.to_tensor(_inputs['input_ids']).unsqueeze(0)
    seg_ids = paddle.to_tensor(_inputs['seg_ids']).unsqueeze(0)
    logits, _ = model(input_ids, seg_ids)
    preds = paddle.argmax(logits, axis=-1).squeeze(0)
    tokens = tokenizer.convert_ids_to_tokens(_inputs['input_ids'][1:seq_len -
                                                                  1])
    labels = preds[1:seq_len - 1].tolist()
    assert len(tokens) == len(labels)
    # add 0 for non punc
    punc_list = [0] + punc_list
    text = ''
    for t, l in zip(tokens, labels):
        text += t
        if l != 0:  # Non punc.
            text += punc_list[l]
    print("Punctuation Restoration Result:", text)
    return text


def main():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(description="Run Punctuation Restoration.")
    parser.add_argument("--config", type=str, help="ErnieLinear config file.")
    parser.add_argument("--checkpoint", type=str, help="snapshot to load.")
    parser.add_argument("--text", type=str, help="raw text to be restored.")
    parser.add_argument("--ngpu",
                        type=int,
                        default=1,
                        help="if ngpu=0, use cpu.")

    args = parser.parse_args()

    if args.ngpu == 0:
        paddle.set_device("cpu")
    elif args.ngpu > 0:
        paddle.set_device("gpu")
    else:
        print("ngpu should >= 0 !")

    test(args)


if __name__ == "__main__":
    main()
