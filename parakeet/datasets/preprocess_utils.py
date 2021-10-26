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
import re


# speaker|utt_id|phn dur phn dur ...
def get_phn_dur(file_name):
    '''
    read MFA duration.txt
    Parameters
    ----------
    file_name : str or Path
        path of gen_duration_from_textgrid.py's result
    Returns
    ----------
    Dict
        sentence: {'utt': ([char], [int])}
    '''
    f = open(file_name, 'r')
    sentence = {}
    speaker_set = set()
    for line in f:
        line_list = line.strip().split('|')
        utt = line_list[0]
        speaker = line_list[1]
        p_d = line_list[-1]
        speaker_set.add(speaker)
        phn_dur = p_d.split()
        phn = phn_dur[::2]
        dur = phn_dur[1::2]
        assert len(phn) == len(dur)
        sentence[utt] = (phn, [int(i) for i in dur], speaker)
    f.close()
    return sentence, speaker_set


def merge_silence(sentence):
    '''
    merge silences
    Parameters
    ----------
    sentence : Dict
        sentence: {'utt': (([char], [int]), str)}
    '''
    for utt in sentence:
        cur_phn, cur_dur, speaker = sentence[utt]
        new_phn = []
        new_dur = []

        # merge sp and sil
        for i, p in enumerate(cur_phn):
            if i > 0 and 'sil' == p and cur_phn[i - 1] in {"sil", "sp"}:
                new_dur[-1] += cur_dur[i]
                new_phn[-1] = 'sil'
            else:
                new_phn.append(p)
                new_dur.append(cur_dur[i])

        for i, (p, d) in enumerate(zip(new_phn, new_dur)):
            if p in {"sp"}:
                if d < 14:
                    new_phn[i] = 'sp'
                else:
                    new_phn[i] = 'spl'

        assert len(new_phn) == len(new_dur)
        sentence[utt] = [new_phn, new_dur, speaker]


def get_input_token(sentence, output_path, dataset="baker"):
    '''
    get phone set from training data and save it
    Parameters
    ----------
    sentence : Dict
        sentence: {'utt': ([char], [int])}
    output_path : str or path
        path to save phone_id_map
    '''
    phn_token = set()
    for utt in sentence:
        for phn in sentence[utt][0]:
            phn_token.add(phn)
    phn_token = list(phn_token)
    phn_token.sort()
    phn_token = ["<pad>", "<unk>"] + phn_token
    if dataset in {"baker", "aishell3"}:
        phn_token += ["，", "。", "？", "！"]
    else:
        phn_token += [",", ".", "?", "!"]
    phn_token += ["<eos>"]

    with open(output_path, 'w') as f:
        for i, phn in enumerate(phn_token):
            f.write(phn + ' ' + str(i) + '\n')


def get_phones_tones(sentence,
                     phones_output_path,
                     tones_output_path,
                     dataset="baker"):
    '''
    get phone set and tone set from training data and save it
    Parameters
    ----------
    sentence : Dict
        sentence: {'utt': ([char], [int])}
    phones_output_path : str or path
        path to save phone_id_map
    tones_output_path : str or path
        path to save tone_id_map
    '''
    phn_token = set()
    tone_token = set()
    for utt in sentence:
        for label in sentence[utt][0]:
            # split tone from finals
            match = re.match(r'^(\w+)([012345])$', label)
            if match:
                phn_token.add(match.group(1))
                tone_token.add(match.group(2))
            else:
                phn_token.add(label)
                tone_token.add('0')
    phn_token = list(phn_token)
    tone_token = list(tone_token)
    phn_token.sort()
    tone_token.sort()
    phn_token = ["<pad>", "<unk>"] + phn_token
    if dataset in {"baker", "aishell3"}:
        phn_token += ["，", "。", "？", "！"]
    else:
        phn_token += [",", ".", "?", "!"]
    phn_token += ["<eos>"]

    with open(phones_output_path, 'w') as f:
        for i, phn in enumerate(phn_token):
            f.write(phn + ' ' + str(i) + '\n')
    with open(tones_output_path, 'w') as f:
        for i, tone in enumerate(tone_token):
            f.write(tone + ' ' + str(i) + '\n')


def get_spk_id_map(speaker_set, output_path):
    speakers = sorted(list(speaker_set))
    with open(output_path, 'w') as f:
        for i, spk in enumerate(speakers):
            f.write(spk + ' ' + str(i) + '\n')


def compare_duration_and_mel_length(sentences, utt, mel):
    '''
    check duration error, correct sentences[utt] if possible, else pop sentences[utt]
    Parameters
    ----------
    sentences : Dict
        sentences[utt] = [phones_list ,durations_list]
    utt : str
        utt_id
    mel : np.ndarry
        features (num_frames, n_mels)
    '''

    if utt in sentences:
        len_diff = mel.shape[0] - sum(sentences[utt][1])
        if len_diff != 0:
            if len_diff > 0:
                sentences[utt][1][-1] += len_diff
            elif sentences[utt][1][-1] + len_diff > 0:
                sentences[utt][1][-1] += len_diff
            elif sentences[utt][1][0] + len_diff > 0:
                sentences[utt][1][0] += len_diff
            else:
                print("the len_diff is unable to correct:", len_diff)
                sentences.pop(utt)
