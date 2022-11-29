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
import os
import re

import paddle
import yaml
from paddlenlp.transformers import ErnieTokenizer
from yacs.config import CfgNode

from paddlespeech.cli.utils import download_and_decompress
from paddlespeech.resource.pretrained_models import rhy_frontend_models
from paddlespeech.text.models.ernie_linear import ErnieLinear
from paddlespeech.utils.env import MODEL_HOME

DefinedClassifier = {
    'ErnieLinear': ErnieLinear,
}

model_version = '1.0'


class Rhy_predictor():
    def __init__(
            self,
            model_dir: os.PathLike=MODEL_HOME, ):
        uncompress_path = download_and_decompress(
            rhy_frontend_models['rhy_e2e'][model_version], model_dir)
        with open(os.path.join(uncompress_path, 'rhy_default.yaml')) as f:
            config = CfgNode(yaml.safe_load(f))
        self.punc_list = []
        with open(os.path.join(uncompress_path, 'rhy_token'), 'r') as f:
            for line in f:
                self.punc_list.append(line.strip())
        self.punc_list = [0] + self.punc_list
        self.make_rhy_dict()
        self.model = DefinedClassifier["ErnieLinear"](**config["model"])
        pretrained_token = config['data_params']['pretrained_token']
        self.tokenizer = ErnieTokenizer.from_pretrained(pretrained_token)
        state_dict = paddle.load(
            os.path.join(uncompress_path, 'snapshot_iter_2600.pdz'))
        self.model.set_state_dict(state_dict["main_params"])
        self.model.eval()

    def _clean_text(self, text):
        text = text.lower()
        text = re.sub('[^A-Za-z0-9\u4e00-\u9fa5]', '', text)
        text = re.sub(f'[{"".join([p for p in self.punc_list][1:])}]', '', text)
        return text

    def preprocess(self, text, tokenizer):
        clean_text = self._clean_text(text)
        assert len(clean_text) > 0, f'Invalid input string: {text}'
        tokenized_input = tokenizer(
            list(clean_text), return_length=True, is_split_into_words=True)
        _inputs = dict()
        _inputs['input_ids'] = tokenized_input['input_ids']
        _inputs['seg_ids'] = tokenized_input['token_type_ids']
        _inputs['seq_len'] = tokenized_input['seq_len']
        return _inputs

    def get_prediction(self, raw_text):
        _inputs = self.preprocess(raw_text, self.tokenizer)
        seq_len = _inputs['seq_len']
        input_ids = paddle.to_tensor(_inputs['input_ids']).unsqueeze(0)
        seg_ids = paddle.to_tensor(_inputs['seg_ids']).unsqueeze(0)
        logits, _ = self.model(input_ids, seg_ids)
        preds = paddle.argmax(logits, axis=-1).squeeze(0)
        tokens = self.tokenizer.convert_ids_to_tokens(
            _inputs['input_ids'][1:seq_len - 1])
        labels = preds[1:seq_len - 1].tolist()
        assert len(tokens) == len(labels)
        # add 0 for non punc
        text = ''
        for t, l in zip(tokens, labels):
            text += t
            if l != 0:  # Non punc.
                text += self.punc_list[l]
        return text

    def make_rhy_dict(self):
        self.rhy_dict = {}
        for i, p in enumerate(self.punc_list[1:]):
            self.rhy_dict[p] = 'sp' + str(i + 1)

    def pinyin_align(self, pinyins, rhy_pre):
        final_py = []
        j = 0
        for i in range(len(rhy_pre)):
            if rhy_pre[i] in self.rhy_dict:
                final_py.append(self.rhy_dict[rhy_pre[i]])
            else:
                final_py.append(pinyins[j])
                j += 1
        return final_py
