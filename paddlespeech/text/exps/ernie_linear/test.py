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

import numpy as np
import paddle
import pandas as pd
import yaml
from paddle import nn
from paddle.io import DataLoader
from paddlespeech.t2s.utils import str2bool
from paddlespeech.text.models.ernie_linear import ErnieLinear
from paddlespeech.text.models.ernie_linear import PuncDataset
from paddlespeech.text.models.ernie_linear import PuncDatasetFromErnieTokenizer
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from yacs.config import CfgNode

DefinedClassifier = {
    'ErnieLinear': ErnieLinear,
}

DefinedLoss = {
    "ce": nn.CrossEntropyLoss,
}

DefinedDataset = {
    'Punc': PuncDataset,
    'Ernie': PuncDatasetFromErnieTokenizer,
}


def evaluation(y_pred, y_test):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=[1, 2, 3])
    overall = precision_recall_fscore_support(
        y_test, y_pred, average='macro', labels=[1, 2, 3])
    result = pd.DataFrame(
        np.array([precision, recall, f1]),
        columns=list(['O', 'COMMA', 'PERIOD', 'QUESTION'])[1:],
        index=['Precision', 'Recall', 'F1'])
    result['OVERALL'] = overall[:3]
    return result


def test(args):
    with open(args.config) as f:
        config = CfgNode(yaml.safe_load(f))
    print("========Args========")
    print(yaml.safe_dump(vars(args)))
    print("========Config========")
    print(config)

    test_dataset = DefinedDataset[config["dataset_type"]](
        train_path=config["test_path"], **config["data_params"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False)
    model = DefinedClassifier[config["model_type"]](**config["model"])
    state_dict = paddle.load(args.checkpoint)
    model.set_state_dict(state_dict["main_params"])
    model.eval()

    punc_list = []
    for i in range(len(test_loader.dataset.id2punc)):
        punc_list.append(test_loader.dataset.id2punc[i])

    test_total_label = []
    test_total_predict = []

    for i, batch in enumerate(test_loader):
        input, label = batch
        label = paddle.reshape(label, shape=[-1])
        y, logit = model(input)
        pred = paddle.argmax(logit, axis=1)
        test_total_label.extend(label.numpy().tolist())
        test_total_predict.extend(pred.numpy().tolist())
    t = classification_report(
        test_total_label, test_total_predict, target_names=punc_list)
    print(t)
    if args.print_eval:
        t2 = evaluation(test_total_label, test_total_predict)
        print('=========================================================')
        print(t2)


def main():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(description="Test a ErnieLinear model.")
    parser.add_argument("--config", type=str, help="ErnieLinear config file.")
    parser.add_argument("--checkpoint", type=str, help="snapshot to load.")
    parser.add_argument("--print_eval", type=str2bool, default=True)
    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu=0, use cpu.")

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
