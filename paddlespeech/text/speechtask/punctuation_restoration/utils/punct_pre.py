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
import shutil

CHINESE_PUNCTUATION_MAPPING = {
    'O': '',
    '，': "，",
    '。': '。',
    '？': '？',
}


def process_one_file_chinese(raw_path, save_path):
    f = open(raw_path, 'r', encoding='utf-8')
    save_file = open(save_path, 'w', encoding='utf-8')
    for line in f.readlines():
        line = line.strip().replace(' ', '').replace(' ', '')
        for i in line:
            save_file.write(i + ' ')
        save_file.write('\n')
    save_file.close()


def process_chinese_pure_senetence(config):
    ####need raw_path, raw_train_file, raw_dev_file, raw_test_file, punc_file, save_path
    assert os.path.exists(
        os.path.join(config["raw_path"], config[
            "raw_train_file"])), "train file doesn't exist."
    assert os.path.exists(
        os.path.join(config["raw_path"], config[
            "raw_dev_file"])), "dev file doesn't exist."
    assert os.path.exists(
        os.path.join(config["raw_path"], config[
            "raw_test_file"])), "test file doesn't exist."
    assert os.path.exists(
        os.path.join(config["raw_path"], config[
            "punc_file"])), "punc file doesn't exist."

    train_file = os.path.join(config["raw_path"], config["raw_train_file"])
    dev_file = os.path.join(config["raw_path"], config["raw_dev_file"])
    test_file = os.path.join(config["raw_path"], config["raw_test_file"])
    if not os.path.exists(config["save_path"]):
        os.makedirs(config["save_path"])

    shutil.copy(
        os.path.join(config["raw_path"], config["punc_file"]),
        os.path.join(config["save_path"], config["punc_file"]))

    process_one_file_chinese(train_file,
                             os.path.join(config["save_path"], "train"))
    process_one_file_chinese(dev_file, os.path.join(config["save_path"], "dev"))
    process_one_file_chinese(test_file,
                             os.path.join(config["save_path"], "test"))


def process_one_chinese_pair(raw_path, save_path):

    f = open(raw_path, 'r', encoding='utf-8')
    save_file = open(save_path, 'w', encoding='utf-8')
    for line in f.readlines():
        if (len(line.strip().split()) == 2):
            word, punc = line.strip().split()
            save_file.write(word + ' ' + CHINESE_PUNCTUATION_MAPPING[punc])
            if (punc == "。"):
                save_file.write("\n")
            else:
                save_file.write(" ")
    save_file.close()


def process_chinese_pair(config):
    ### need raw_path, raw_train_file, raw_dev_file, raw_test_file, punc_file, save_path
    assert os.path.exists(
        os.path.join(config["raw_path"], config[
            "raw_train_file"])), "train file doesn't exist."
    assert os.path.exists(
        os.path.join(config["raw_path"], config[
            "raw_dev_file"])), "dev file doesn't exist."
    assert os.path.exists(
        os.path.join(config["raw_path"], config[
            "raw_test_file"])), "test file doesn't exist."
    assert os.path.exists(
        os.path.join(config["raw_path"], config[
            "punc_file"])), "punc file doesn't exist."

    train_file = os.path.join(config["raw_path"], config["raw_train_file"])
    dev_file = os.path.join(config["raw_path"], config["raw_dev_file"])
    test_file = os.path.join(config["raw_path"], config["raw_test_file"])

    process_one_chinese_pair(train_file,
                             os.path.join(config["save_path"], "train"))
    process_one_chinese_pair(dev_file, os.path.join(config["save_path"], "dev"))
    process_one_chinese_pair(test_file,
                             os.path.join(config["save_path"], "test"))

    shutil.copy(
        os.path.join(config["raw_path"], config["punc_file"]),
        os.path.join(config["save_path"], config["punc_file"]))


english_punc = [',', '.', '?']
ignore_english_punc = ['\"', '/']


def process_one_file_english(raw_path, save_path):
    f = open(raw_path, 'r', encoding='utf-8')
    save_file = open(save_path, 'w', encoding='utf-8')
    for line in f.readlines():
        for i in ignore_english_punc:
            line = line.replace(i, '')
        for i in english_punc:
            line = line.replace(i, ' ' + i)
        wordlist = line.strip().split(' ')
        # print(type(wordlist))
        # print(wordlist)
        for i in wordlist:
            save_file.write(i + ' ')
        save_file.write('\n')
    save_file.close()


def process_english_pure_senetence(config):
    ####need raw_path, raw_train_file, raw_dev_file, raw_test_file, punc_file, save_path
    assert os.path.exists(
        os.path.join(config["raw_path"], config[
            "raw_train_file"])), "train file doesn't exist."
    assert os.path.exists(
        os.path.join(config["raw_path"], config[
            "raw_dev_file"])), "dev file doesn't exist."
    assert os.path.exists(
        os.path.join(config["raw_path"], config[
            "raw_test_file"])), "test file doesn't exist."
    assert os.path.exists(
        os.path.join(config["raw_path"], config[
            "punc_file"])), "punc file doesn't exist."

    train_file = os.path.join(config["raw_path"], config["raw_train_file"])
    dev_file = os.path.join(config["raw_path"], config["raw_dev_file"])
    test_file = os.path.join(config["raw_path"], config["raw_test_file"])
    if not os.path.exists(config["save_path"]):
        os.makedirs(config["save_path"])

    shutil.copy(
        os.path.join(config["raw_path"], config["punc_file"]),
        os.path.join(config["save_path"], config["punc_file"]))

    process_one_file_english(train_file,
                             os.path.join(config["save_path"], "train"))
    process_one_file_english(dev_file, os.path.join(config["save_path"], "dev"))
    process_one_file_english(test_file,
                             os.path.join(config["save_path"], "test"))
