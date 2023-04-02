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
import codecs
import os


# org_split = 'train-split/train-segment'
# text_file = 'En-Zh/train.en-zh'
# data_split = 'train'
def data_process(src_dir, tgt_dir, wav_dir_list, text_file_list,
                 data_split_list):

    for org_split, text_file, data_split in zip(wav_dir_list, text_file_list,
                                                data_split_list):
        local_data_split_dir = os.path.join(tgt_dir, data_split)

        os.makedirs(local_data_split_dir, exist_ok=True)
        utts = []
        utt2spk = {}
        with open(os.path.join(local_data_split_dir, 'wav.scp.org'), 'w') as wav_wf, \
            open(os.path.join(local_data_split_dir, 'utt2spk.org'), 'w') as utt2spk_wf:
            for files in os.listdir(os.path.join(src_dir, org_split)):
                files = files.strip()
                file_path = os.path.join(src_dir, org_split, files)
                size = os.path.getsize(file_path)
                if size <= 30000:
                    continue
                utt = files.split('.')[0]
                audio_name = utt.split('_')[0]
                #format the name of utterance
                while len(audio_name) < 6:
                    utt = '0' + utt
                    audio_name = '0' + audio_name
                utt = 'ted-en-zh-' + utt
                utts.append(utt)
                spk = utt.split('_')[0]
                utt2spk[utt] = spk
                assert len(spk) == 16, "%r" % spk
                print(utt, 'cat', os.path.abspath(file_path), '|', file=wav_wf)
            for utt in sorted(utts):
                print(utt, utt2spk[utt], file=utt2spk_wf)

        with open(os.path.join(local_data_split_dir, 'en.org'), 'w') as en_wf, \
            open(os.path.join(local_data_split_dir, 'zh.org'), 'w') as zh_wf, \
            open(os.path.join(local_data_split_dir, '.yaml'), 'w') as yaml_wf, \
            codecs.open(os.path.join(src_dir, text_file), 'r', encoding='utf-8',
                        errors='ignore') as rf:
            count = 0
            for line in rf:
                line = line.strip()
                line_spl = line.split('\t')
                assert len(line_spl) == 3, "%r" % line
                wav, en, zh = line_spl
                assert wav.endswith('wav'), "%r" % wav[-3:]
                utt = wav.split('.')[0]
                audio_name = utt.split('_')[0]
                while len(audio_name) < 6:
                    utt = '0' + utt
                    audio_name = '0' + audio_name
                utt = 'ted-en-zh-' + utt
                print(utt, file=yaml_wf)
                print(en.lower(), file=en_wf)
                print(zh, file=zh_wf)
                count += 1
            print('%s set lines count: %d' % (data_split, count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--src-dir",
        default="",
        type=str,
        help="Directory to kaldi splited data. (default: %(default)s)")
    parser.add_argument(
        "--tgt-dir",
        default="local/ted_en_zh",
        type=str,
        help="Directory to save processed data. (default: %(default)s)")
    args = parser.parse_args()

    wav_dir_list = [
        'train-split/train-segment', 'test-segment/tst2014',
        'test-segment/tst2015'
    ]
    text_file_list = [
        'En-Zh/train.en-zh', 'En-Zh/tst2014.en-zh', 'En-Zh/tst2015.en-zh'
    ]
    data_split_list = ['train', 'dev', 'test']
    data_process(args.src_dir, args.tgt_dir, wav_dir_list, text_file_list,
                 data_split_list)
