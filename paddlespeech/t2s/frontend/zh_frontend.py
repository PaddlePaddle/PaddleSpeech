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
from operator import itemgetter
from typing import Dict
from typing import List

import jieba.posseg as psg
import numpy as np
import paddle
import yaml
from g2pM import G2pM
from pypinyin import lazy_pinyin
from pypinyin import load_phrases_dict
from pypinyin import load_single_dict
from pypinyin import Style
from pypinyin_dict.phrase_pinyin_data import large_pinyin

from paddlespeech.t2s.frontend.g2pw import G2PWOnnxConverter
from paddlespeech.t2s.frontend.generate_lexicon import generate_lexicon
from paddlespeech.t2s.frontend.rhy_prediction.rhy_predictor import RhyPredictor
from paddlespeech.t2s.frontend.ssml.xml_processor import MixTextProcessor
from paddlespeech.t2s.frontend.tone_sandhi import ToneSandhi
from paddlespeech.t2s.frontend.zh_normalization.text_normlization import TextNormalizer

INITIALS = [
    'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'zh', 'ch', 'sh',
    'r', 'z', 'c', 's', 'j', 'q', 'x'
]
INITIALS += ['y', 'w', 'sp', 'spl', 'spn', 'sil']


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def insert_after_character(lst, item):
    """
    inset `item` after finals.
    """
    result = [item]

    for phone in lst:
        result.append(phone)
        if phone not in INITIALS:
            # finals has tones
            # assert phone[-1] in "12345"
            result.append(item)

    return result


class Polyphonic():
    def __init__(self):
        with open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'polyphonic.yaml'),
                'r',
                encoding='utf-8') as polyphonic_file:
            # 解析yaml
            polyphonic_dict = yaml.load(polyphonic_file, Loader=yaml.FullLoader)
        self.polyphonic_words = polyphonic_dict["polyphonic"]

    def correct_pronunciation(self, word, pinyin):
        # 词汇被词典收录则返回纠正后的读音
        if word in self.polyphonic_words.keys():
            pinyin = self.polyphonic_words[word]
        # 否则返回原读音
        return pinyin


class Frontend():
    def __init__(self,
                 g2p_model="g2pW",
                 phone_vocab_path=None,
                 tone_vocab_path=None,
                 use_rhy=False):

        self.punc = "：，；。？！“”‘’':,;.?!"
        self.rhy_phns = ['sp1', 'sp2', 'sp3', 'sp4']
        self.phrases_dict = {
            '开户行': [['ka1i'], ['hu4'], ['hang2']],
            '发卡行': [['fa4'], ['ka3'], ['hang2']],
            '放款行': [['fa4ng'], ['kua3n'], ['hang2']],
            '茧行': [['jia3n'], ['hang2']],
            '行号': [['hang2'], ['ha4o']],
            '各地': [['ge4'], ['di4']],
            '借还款': [['jie4'], ['hua2n'], ['kua3n']],
            '时间为': [['shi2'], ['jia1n'], ['we2i']],
            '为准': [['we2i'], ['zhu3n']],
            '色差': [['se4'], ['cha1']],
            '嗲': [['dia3']],
            '呗': [['bei5']],
            '不': [['bu4']],
            '咗': [['zuo5']],
            '嘞': [['lei5']],
            '掺和': [['chan1'], ['huo5']]
        }

        self.must_erhua = {
            "小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿", "媳妇儿"
        }
        self.not_erhua = {
            "虐儿", "为儿", "护儿", "瞒儿", "救儿", "替儿", "有儿", "一儿", "我儿", "俺儿", "妻儿",
            "拐儿", "聋儿", "乞儿", "患儿", "幼儿", "孤儿", "婴儿", "婴幼儿", "连体儿", "脑瘫儿",
            "流浪儿", "体弱儿", "混血儿", "蜜雪儿", "舫儿", "祖儿", "美儿", "应采儿", "可儿", "侄儿",
            "孙儿", "侄孙儿", "女儿", "男儿", "红孩儿", "花儿", "虫儿", "马儿", "鸟儿", "猪儿", "猫儿",
            "狗儿", "少儿"
        }

        self.vocab_phones = {}
        self.vocab_tones = {}
        if phone_vocab_path:
            with open(phone_vocab_path, 'rt', encoding='utf-8') as f:
                phn_id = [line.strip().split() for line in f.readlines()]
            for phn, id in phn_id:
                self.vocab_phones[phn] = int(id)
        if tone_vocab_path:
            with open(tone_vocab_path, 'rt', encoding='utf-8') as f:
                tone_id = [line.strip().split() for line in f.readlines()]
            for tone, id in tone_id:
                self.vocab_tones[tone] = int(id)

        # SSML
        self.mix_ssml_processor = MixTextProcessor()
        # tone sandhi
        self.tone_modifier = ToneSandhi()
        # TN
        self.text_normalizer = TextNormalizer()

        # prosody
        self.use_rhy = use_rhy
        if use_rhy:
            self.rhy_predictor = RhyPredictor()
            print("Rhythm predictor loaded.")

        # g2p
        assert g2p_model in ('pypinyin', 'g2pM', 'g2pW')
        self.g2p_model = g2p_model
        if self.g2p_model == "g2pM":
            self.g2pM_model = G2pM()
            self.pinyin2phone = generate_lexicon(
                with_tone=True, with_erhua=False)
        elif self.g2p_model == "g2pW":
            # use pypinyin as backup for non polyphonic characters in g2pW
            self._init_pypinyin()
            self.corrector = Polyphonic()
            self.g2pM_model = G2pM()
            self.g2pW_model = G2PWOnnxConverter(
                style='pinyin', enable_non_tradional_chinese=True)
            self.pinyin2phone = generate_lexicon(
                with_tone=True, with_erhua=False)
        else:
            self._init_pypinyin()

    def _init_pypinyin(self):
        """
        Load pypinyin G2P module.
        """
        large_pinyin.load()
        load_phrases_dict(self.phrases_dict)
        # 调整字的拼音顺序
        load_single_dict({ord(u'地'): u'de,di4'})

    def _get_initials_finals(self, word: str) -> List[List[str]]:
        """
        Get word initial and final by pypinyin or g2pM
        """
        initials = []
        finals = []
        if self.g2p_model == "pypinyin":
            orig_initials = lazy_pinyin(
                word, neutral_tone_with_five=True, style=Style.INITIALS)
            orig_finals = lazy_pinyin(
                word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
            for c, v in zip(orig_initials, orig_finals):
                if re.match(r'i\d', v):
                    if c in ['z', 'c', 's']:
                        # zi, ci, si
                        v = re.sub('i', 'ii', v)
                    elif c in ['zh', 'ch', 'sh', 'r']:
                        # zhi, chi, shi
                        v = re.sub('i', 'iii', v)
                initials.append(c)
                finals.append(v)

        elif self.g2p_model == "g2pM":
            pinyins = self.g2pM_model(word, tone=True, char_split=False)
            for pinyin in pinyins:
                pinyin = pinyin.replace("u:", "v")
                if pinyin in self.pinyin2phone:
                    initial_final_list = self.pinyin2phone[pinyin].split(" ")
                    if len(initial_final_list) == 2:
                        initials.append(initial_final_list[0])
                        finals.append(initial_final_list[1])
                    elif len(initial_final_list) == 1:
                        initials.append('')
                        finals.append(initial_final_list[1])
                else:
                    # If it's not pinyin (possibly punctuation) or no conversion is required
                    initials.append(pinyin)
                    finals.append(pinyin)

        return initials, finals

    def _merge_erhua(self,
                     initials: List[str],
                     finals: List[str],
                     word: str,
                     pos: str) -> List[List[str]]:
        """
        Do erhub.
        """
        # fix er1
        for i, phn in enumerate(finals):
            if i == len(finals) - 1 and word[i] == "儿" and phn == 'er1':
                finals[i] = 'er2'

        # 发音
        if word not in self.must_erhua and (word in self.not_erhua or
                                            pos in {"a", "j", "nr"}):
            return initials, finals

        # "……" 等情况直接返回
        if len(finals) != len(word):
            return initials, finals

        assert len(finals) == len(word)

        # 不发音
        new_initials = []
        new_finals = []
        for i, phn in enumerate(finals):
            if i == len(finals) - 1 and word[i] == "儿" and phn in {
                    "er2", "er5"
            } and word[-2:] not in self.not_erhua and new_finals:
                new_finals[-1] = new_finals[-1][:-1] + "r" + new_finals[-1][-1]
            else:
                new_initials.append(initials[i])
                new_finals.append(phn)

        return new_initials, new_finals

    # if merge_sentences, merge all sentences into one phone sequence
    def _g2p(self,
             sentences: List[str],
             merge_sentences: bool=True,
             with_erhua: bool=True) -> List[List[str]]:
        """
        Return: list of list phonemes.
            [['w', 'o3', 'm', 'en2', 'sp'], ...]
        """
        segments = sentences
        phones_list = []

        # split by punctuation
        for seg in segments:
            if self.use_rhy:
                seg = self.rhy_predictor._clean_text(seg)

            # remove all English words in the sentence
            seg = re.sub('[a-zA-Z]+', '', seg)

            # add prosody mark
            if self.use_rhy:
                seg = self.rhy_predictor.get_prediction(seg)

            # [(word, pos), ...]
            seg_cut = psg.lcut(seg)
            # fix wordseg bad case for sandhi
            seg_cut = self.tone_modifier.pre_merge_for_modify(seg_cut)

            # 为了多音词获得更好的效果，这里采用整句预测
            phones = []
            initials = []
            finals = []
            if self.g2p_model == "g2pW":
                try:
                    # undo prosody 
                    if self.use_rhy:
                        seg = self.rhy_predictor._clean_text(seg)

                    # g2p
                    pinyins = self.g2pW_model(seg)[0]
                except Exception:
                    # g2pW 模型采用繁体输入，如果有cover不了的简体词，采用g2pM预测
                    print("[%s] not in g2pW dict,use g2pM" % seg)
                    pinyins = self.g2pM_model(seg, tone=True, char_split=False)

                # do prosody
                if self.use_rhy:
                    rhy_text = self.rhy_predictor.get_prediction(seg)
                    final_py = self.rhy_predictor.pinyin_align(pinyins,
                                                               rhy_text)
                    pinyins = final_py

                pre_word_length = 0
                for word, pos in seg_cut:
                    sub_initials = []
                    sub_finals = []
                    now_word_length = pre_word_length + len(word)

                    # skip english word
                    if pos == 'eng':
                        pre_word_length = now_word_length
                        continue

                    word_pinyins = pinyins[pre_word_length:now_word_length]

                    # 多音字消歧
                    word_pinyins = self.corrector.correct_pronunciation(
                        word, word_pinyins)

                    for pinyin, char in zip(word_pinyins, word):
                        if pinyin is None:
                            pinyin = char

                        pinyin = pinyin.replace("u:", "v")

                        if pinyin in self.pinyin2phone:
                            initial_final_list = self.pinyin2phone[
                                pinyin].split(" ")
                            if len(initial_final_list) == 2:
                                sub_initials.append(initial_final_list[0])
                                sub_finals.append(initial_final_list[1])
                            elif len(initial_final_list) == 1:
                                sub_initials.append('')
                                sub_finals.append(initial_final_list[1])
                        else:
                            # If it's not pinyin (possibly punctuation) or no conversion is required
                            sub_initials.append(pinyin)
                            sub_finals.append(pinyin)

                    pre_word_length = now_word_length
                    # tone sandhi
                    sub_finals = self.tone_modifier.modified_tone(word, pos,
                                                                  sub_finals)
                    # er hua                                
                    if with_erhua:
                        sub_initials, sub_finals = self._merge_erhua(
                            sub_initials, sub_finals, word, pos)

                    initials.append(sub_initials)
                    finals.append(sub_finals)
                    # assert len(sub_initials) == len(sub_finals) == len(word)
            else:
                # pypinyin, g2pM
                for word, pos in seg_cut:
                    if pos == 'eng':
                        # skip english word
                        continue

                    # g2p
                    sub_initials, sub_finals = self._get_initials_finals(word)
                    # tone sandhi
                    sub_finals = self.tone_modifier.modified_tone(word, pos,
                                                                  sub_finals)
                    # er hua
                    if with_erhua:
                        sub_initials, sub_finals = self._merge_erhua(
                            sub_initials, sub_finals, word, pos)

                    initials.append(sub_initials)
                    finals.append(sub_finals)
                    # assert len(sub_initials) == len(sub_finals) == len(word)

                # sum(iterable[, start])
            initials = sum(initials, [])
            finals = sum(finals, [])

            for c, v in zip(initials, finals):
                # NOTE: post process for pypinyin outputs
                # we discriminate i, ii and iii
                if c and c not in self.punc:
                    phones.append(c)
                # replace punctuation by `sp`
                if c and c in self.punc:
                    phones.append('sp')

                if v and v not in self.punc and v not in self.rhy_phns:
                    phones.append(v)

            phones_list.append(phones)

        # merge split sub sentence into one sentence.
        if merge_sentences:
            # sub sentence phonemes
            merge_list = sum(phones_list, [])
            # rm the last 'sp' to avoid the noise at the end
            # cause in the training data, no 'sp' in the end
            if merge_list[-1] == 'sp':
                merge_list = merge_list[:-1]

            # sentence phonemes
            phones_list = []
            phones_list.append(merge_list)

        return phones_list

    def _p2id(self, phonemes: List[str]) -> np.ndarray:
        """
        Phoneme to Index
        """
        # replace unk phone with sp
        phonemes = [
            phn if phn in self.vocab_phones else "sp" for phn in phonemes
        ]
        phone_ids = [self.vocab_phones[item] for item in phonemes]
        return np.array(phone_ids, np.int64)

    def _t2id(self, tones: List[str]) -> np.ndarray:
        """
        Tone to Index.
        """
        # replace unk phone with sp
        tones = [tone if tone in self.vocab_tones else "0" for tone in tones]
        tone_ids = [self.vocab_tones[item] for item in tones]
        return np.array(tone_ids, np.int64)

    def _get_phone_tone(self, phonemes: List[str],
                        get_tone_ids: bool=False) -> List[List[str]]:
        """
        Get tone from phonemes.
        """
        phones = []
        tones = []
        if get_tone_ids and self.vocab_tones:
            for full_phone in phonemes:
                # split tone from finals
                match = re.match(r'^(\w+)([012345])$', full_phone)
                if match:
                    phone = match.group(1)
                    tone = match.group(2)
                    # if the merged erhua not in the vocab
                    # assume that the input is ['iaor3'] and 'iaor' not in self.vocab_phones, we split 'iaor' into ['iao','er']
                    # and the tones accordingly change from ['3'] to ['3','2'], while '2' is the tone of 'er2'
                    if len(phone) >= 2 and phone != "er" and phone[
                            -1] == 'r' and phone not in self.vocab_phones and phone[:
                                                                                    -1] in self.vocab_phones:
                        phones.append(phone[:-1])
                        tones.append(tone)
                        phones.append("er")
                        tones.append("2")
                    else:
                        phones.append(phone)
                        tones.append(tone)
                else:
                    # initals with 0 tone.
                    phones.append(full_phone)
                    tones.append('0')
        else:
            for phone in phonemes:
                # if the merged erhua not in the vocab
                # assume that the input is ['iaor3'] and 'iaor' not in self.vocab_phones, change ['iaor3'] to ['iao3','er2']
                if len(phone) >= 3 and phone[:-1] != "er" and phone[
                        -2] == 'r' and phone not in self.vocab_phones and (
                            phone[:-2] + phone[-1]) in self.vocab_phones:
                    phones.append((phone[:-2] + phone[-1]))
                    phones.append("er2")
                else:
                    phones.append(phone)

        return phones, tones

    def get_phonemes(self,
                     sentence: str,
                     merge_sentences: bool=True,
                     with_erhua: bool=True,
                     robot: bool=False,
                     print_info: bool=False) -> List[List[str]]:
        """
        Main function to do G2P
        """
        # TN & Text Segmentation
        sentences = self.text_normalizer.normalize(sentence)
        # Prosody & WS & g2p & tone sandhi
        phonemes = self._g2p(
            sentences, merge_sentences=merge_sentences, with_erhua=with_erhua)

        # simulate robot pronunciation, change all tones to `1`
        if robot:
            new_phonemes = []
            for sentence in phonemes:
                new_sentence = []
                for item in sentence:
                    # `er` only have tone `2`
                    if item[-1] in "12345" and item != "er2":
                        item = item[:-1] + "1"
                    new_sentence.append(item)
                new_phonemes.append(new_sentence)
            phonemes = new_phonemes

        if print_info:
            print("----------------------------")
            print("text norm results:")
            print(sentences)
            print("----------------------------")
            print("g2p results:")
            print(phonemes)
            print("----------------------------")
        return phonemes

    def _split_word_to_char(self, words):
        res = []
        for x in words:
            res.append(x)
        return res

    # if using ssml, have pingyin specified, assign pinyin to words
    def _g2p_assign(self,
                    words: List[str],
                    pinyin_spec: List[str],
                    merge_sentences: bool=True) -> List[List[str]]:
        """
        Replace phoneme by SSML
        """
        phones_list = []
        initials = []
        finals = []

        # to charactor list
        words = self._split_word_to_char(words[0])

        for pinyin, char in zip(pinyin_spec, words):
            sub_initials = []
            sub_finals = []
            pinyin = pinyin.replace("u:", "v")

            #self.pinyin2phone: is a dict with all pinyin mapped with sheng_mu yun_mu
            if pinyin in self.pinyin2phone:
                initial_final_list = self.pinyin2phone[pinyin].split(" ")
                if len(initial_final_list) == 2:
                    sub_initials.append(initial_final_list[0])
                    sub_finals.append(initial_final_list[1])
                elif len(initial_final_list) == 1:
                    sub_initials.append('')
                    sub_finals.append(initial_final_list[1])
            else:
                # If it's not pinyin (possibly punctuation) or no conversion is required
                sub_initials.append(pinyin)
                sub_finals.append(pinyin)

            initials.append(sub_initials)
            finals.append(sub_finals)

        initials = sum(initials, [])
        finals = sum(finals, [])

        phones = []
        for c, v in zip(initials, finals):
            # NOTE: post process for pypinyin outputs
            # we discriminate i, ii and iii
            if c and c not in self.punc:
                phones.append(c)
            # replace punc to `sp`
            if c and c in self.punc:
                phones.append('sp')
            if v and v not in self.punc and v not in self.rhy_phns:
                phones.append(v)
        phones_list.append(phones)

        if merge_sentences:
            merge_list = sum(phones_list, [])
            # rm the last 'sp' to avoid the noise at the end
            # cause in the training data, no 'sp' in the end
            if merge_list[-1] == 'sp':
                merge_list = merge_list[:-1]
            phones_list = []
            phones_list.append(merge_list)

        return phones_list

    def get_phonemes_ssml(self,
                          ssml_inputs: list,
                          merge_sentences: bool=True,
                          with_erhua: bool=True,
                          robot: bool=False,
                          print_info: bool=False) -> List[List[str]]:
        """
         Main function to do G2P with SSML support.
        """
        all_phonemes = []
        for word_pinyin_item in ssml_inputs:
            phonemes = []
            print("ssml inputs:", word_pinyin_item)
            sentence, pinyin_spec = itemgetter(0, 1)(word_pinyin_item)
            print('ssml g2p:', sentence, pinyin_spec)
            # TN & Text Segmentation
            sentences = self.text_normalizer.normalize(sentence)
            if len(pinyin_spec) == 0:
                # g2p word w/o specified <say-as>
                phonemes = self._g2p(
                    sentences,
                    merge_sentences=merge_sentences,
                    with_erhua=with_erhua)
            else:
                # word phonemes specified by <say-as>
                phonemes = self._g2p_assign(
                    sentences, pinyin_spec, merge_sentences=merge_sentences)

            all_phonemes = all_phonemes + phonemes

        if robot:
            new_phonemes = []
            for sentence in all_phonemes:
                new_sentence = []
                for item in sentence:
                    # `er` only have tone `2`
                    if item[-1] in "12345" and item != "er2":
                        item = item[:-1] + "1"
                    new_sentence.append(item)
                new_phonemes.append(new_sentence)
            all_phonemes = new_phonemes

        if print_info:
            print("----------------------------")
            print("text norm results:")
            print(sentences)
            print("----------------------------")
            print("g2p results:")
            print(all_phonemes[0])
            print("----------------------------")
        return [sum(all_phonemes, [])]

    def add_sp_if_no(self, phonemes):
        """
        Prosody mark #4 added at sentence end.
        """
        if not phonemes[-1][-1].startswith('sp'):
            phonemes[-1].append('sp4')
        return phonemes

    def get_input_ids(self,
                      sentence: str,
                      merge_sentences: bool=True,
                      get_tone_ids: bool=False,
                      robot: bool=False,
                      print_info: bool=False,
                      add_blank: bool=False,
                      blank_token: str="<pad>",
                      to_tensor: bool=True) -> Dict[str, List[paddle.Tensor]]:

        phonemes = self.get_phonemes(
            sentence,
            merge_sentences=merge_sentences,
            print_info=print_info,
            robot=robot)

        # add #4 for sentence end.
        if self.use_rhy:
            phonemes = self.add_sp_if_no(phonemes)

        result = {}
        phones = []
        tones = []
        temp_phone_ids = []
        temp_tone_ids = []

        for part_phonemes in phonemes:

            phones, tones = self._get_phone_tone(
                part_phonemes, get_tone_ids=get_tone_ids)

            if add_blank:
                phones = insert_after_character(phones, blank_token)

            if tones:
                tone_ids = self._t2id(tones)
                if to_tensor:
                    tone_ids = paddle.to_tensor(tone_ids)
                temp_tone_ids.append(tone_ids)

            if phones:
                phone_ids = self._p2id(phones)
                # if use paddle.to_tensor() in onnxruntime, the first time will be too low
                if to_tensor:
                    phone_ids = paddle.to_tensor(phone_ids)
                temp_phone_ids.append(phone_ids)

        if temp_tone_ids:
            result["tone_ids"] = temp_tone_ids
        if temp_phone_ids:
            result["phone_ids"] = temp_phone_ids

        return result

    def get_input_ids_ssml(
            self,
            sentence: str,
            merge_sentences: bool=True,
            get_tone_ids: bool=False,
            robot: bool=False,
            print_info: bool=False,
            add_blank: bool=False,
            blank_token: str="<pad>",
            to_tensor: bool=True) -> Dict[str, List[paddle.Tensor]]:

        # split setence by SSML tag.
        l_inputs = MixTextProcessor.get_pinyin_split(sentence)

        phonemes = self.get_phonemes_ssml(
            l_inputs,
            merge_sentences=merge_sentences,
            print_info=print_info,
            robot=robot)

        result = {}
        phones = []
        tones = []
        temp_phone_ids = []
        temp_tone_ids = []

        for part_phonemes in phonemes:
            phones, tones = self._get_phone_tone(
                part_phonemes, get_tone_ids=get_tone_ids)

            if add_blank:
                phones = insert_after_character(phones, blank_token)

            if tones:
                tone_ids = self._t2id(tones)
                if to_tensor:
                    tone_ids = paddle.to_tensor(tone_ids)
                temp_tone_ids.append(tone_ids)

            if phones:
                phone_ids = self._p2id(phones)
                # if use paddle.to_tensor() in onnxruntime, the first time will be too low
                if to_tensor:
                    phone_ids = paddle.to_tensor(phone_ids)
                temp_phone_ids.append(phone_ids)

        if temp_tone_ids:
            result["tone_ids"] = temp_tone_ids
        if temp_phone_ids:
            result["phone_ids"] = temp_phone_ids

        return result
