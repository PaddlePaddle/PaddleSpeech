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
from typing import List

import librosa
import numpy as np


# speaker|utt_id|phn dur phn dur ...
def get_phn_dur(file_name):
    '''
    read MFA duration.txt
    Args:
        file_name (str or Path): path of gen_duration_from_textgrid.py's result
    Returns: 
        Dict: sentence: {'utt': ([char], [int])}
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


def note2midi(notes: List[str]) -> List[str]:
    """Covert note string to note id, for example: ["C1"] -> [24]

    Args:
        notes (List[str]): the list of note string

    Returns:
        List[str]: the list of note id
    """
    midis = []
    for note in notes:
        if note == 'rest':
            midi = 0
        else:
            midi = librosa.note_to_midi(note.split("/")[0])
        midis.append(midi)

    return midis


def time2frame(
        times: List[float],
        sample_rate: int=24000,
        n_shift: int=128, ) -> List[int]:
    """Convert the phoneme duration of time(s) into frames

    Args:
        times (List[float]): phoneme duration of time(s)
        sample_rate (int, optional): sample rate. Defaults to 24000.
        n_shift (int, optional): frame shift. Defaults to 128.

    Returns:
        List[int]: phoneme duration of frame
    """
    end = 0.0
    ends = []
    for t in times:
        end += t
        ends.append(end)
    frame_pos = librosa.time_to_frames(ends, sr=sample_rate, hop_length=n_shift)
    durations = np.diff(frame_pos, prepend=0)
    return durations


def get_sentences_svs(
        file_name,
        dataset: str='opencpop',
        sample_rate: int=24000,
        n_shift: int=128, ):
    '''
    read label file
    Args:
        file_name (str or Path): path of gen_duration_from_textgrid.py's result
        dataset (str): dataset name
    Returns: 
        Dict: the information of sentence, include [phone id (int)], [the frame of phone (int)], [note id (int)], [note duration (float)], [is slur (int)], text(str), speaker name (str)
        tuple: speaker name
    '''
    f = open(file_name, 'r')
    sentence = {}
    speaker_set = set()
    if dataset == 'opencpop':
        speaker_set.add("opencpop")
        for line in f:
            line_list = line.strip().split('|')
            utt = line_list[0]
            text = line_list[1]
            ph = line_list[2].split()
            midi = note2midi(line_list[3].split())
            midi_dur = line_list[4].split()
            ph_dur = time2frame([float(t) for t in line_list[5].split()], sample_rate=sample_rate, n_shift=n_shift)
            is_slur = line_list[6].split()
            assert len(ph) == len(midi) == len(midi_dur) == len(is_slur)
            sentence[utt] = (ph, [int(i) for i in ph_dur],
                             [int(i) for i in midi],
                             [float(i) for i in midi_dur],
                             [int(i) for i in is_slur], text, "opencpop")
    else:
        print("dataset should in {opencpop} now!")

    f.close()
    return sentence, speaker_set


def merge_silence(sentence):
    '''
    merge silences
    Args:
        sentence (Dict): sentence: {'utt': (([char], [int]), str)}
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
    Args:
        sentence (Dict): sentence: {'utt': ([char], [int])}
        output_path (str or path):path to save phone_id_map
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
    # svs dataset
    elif dataset in {"opencpop"}:
        pass
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
    Args:
        sentence (Dict): sentence: {'utt': ([char], [int])}
        phones_output_path (str or path): path to save phone_id_map
        tones_output_path (str or path): path to save tone_id_map
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
    Args:
        sentences (Dict): sentences[utt] = [phones_list ,durations_list]
        utt (str): utt_id
        mel (np.ndarry): features (num_frames, n_mels)
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
