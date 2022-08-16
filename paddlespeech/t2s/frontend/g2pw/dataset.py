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
"""
Credits
    This code is modified from https://github.com/GitYCC/g2pW
"""
import numpy as np

from paddlespeech.t2s.frontend.g2pw.utils import tokenize_and_map

ANCHOR_CHAR = '▁'


def prepare_onnx_input(tokenizer,
                       labels,
                       char2phonemes,
                       chars,
                       texts,
                       query_ids,
                       phonemes=None,
                       pos_tags=None,
                       use_mask=False,
                       use_char_phoneme=False,
                       use_pos=False,
                       window_size=None,
                       max_len=512):
    if window_size is not None:
        truncated_texts, truncated_query_ids = _truncate_texts(window_size,
                                                               texts, query_ids)

    input_ids = []
    token_type_ids = []
    attention_masks = []
    phoneme_masks = []
    char_ids = []
    position_ids = []

    for idx in range(len(texts)):
        text = (truncated_texts if window_size else texts)[idx].lower()
        query_id = (truncated_query_ids if window_size else query_ids)[idx]

        try:
            tokens, text2token, token2text = tokenize_and_map(tokenizer, text)
        except Exception:
            print(f'warning: text "{text}" is invalid')
            return {}

        text, query_id, tokens, text2token, token2text = _truncate(
            max_len, text, query_id, tokens, text2token, token2text)

        processed_tokens = ['[CLS]'] + tokens + ['[SEP]']

        input_id = list(
            np.array(tokenizer.convert_tokens_to_ids(processed_tokens)))
        token_type_id = list(np.zeros((len(processed_tokens), ), dtype=int))
        attention_mask = list(np.ones((len(processed_tokens), ), dtype=int))

        query_char = text[query_id]
        phoneme_mask = [1 if i in char2phonemes[query_char] else 0 for i in range(len(labels))] \
            if use_mask else [1] * len(labels)
        char_id = chars.index(query_char)
        position_id = text2token[
            query_id] + 1  # [CLS] token locate at first place

        input_ids.append(input_id)
        token_type_ids.append(token_type_id)
        attention_masks.append(attention_mask)
        phoneme_masks.append(phoneme_mask)
        char_ids.append(char_id)
        position_ids.append(position_id)

    outputs = {
        'input_ids': np.array(input_ids),
        'token_type_ids': np.array(token_type_ids),
        'attention_masks': np.array(attention_masks),
        'phoneme_masks': np.array(phoneme_masks).astype(np.float32),
        'char_ids': np.array(char_ids),
        'position_ids': np.array(position_ids),
    }
    return outputs


def _truncate_texts(window_size, texts, query_ids):
    truncated_texts = []
    truncated_query_ids = []
    for text, query_id in zip(texts, query_ids):
        start = max(0, query_id - window_size // 2)
        end = min(len(text), query_id + window_size // 2)
        truncated_text = text[start:end]
        truncated_texts.append(truncated_text)

        truncated_query_id = query_id - start
        truncated_query_ids.append(truncated_query_id)
    return truncated_texts, truncated_query_ids


def _truncate(max_len, text, query_id, tokens, text2token, token2text):
    truncate_len = max_len - 2
    if len(tokens) <= truncate_len:
        return (text, query_id, tokens, text2token, token2text)

    token_position = text2token[query_id]

    token_start = token_position - truncate_len // 2
    token_end = token_start + truncate_len
    font_exceed_dist = -token_start
    back_exceed_dist = token_end - len(tokens)
    if font_exceed_dist > 0:
        token_start += font_exceed_dist
        token_end += font_exceed_dist
    elif back_exceed_dist > 0:
        token_start -= back_exceed_dist
        token_end -= back_exceed_dist

    start = token2text[token_start][0]
    end = token2text[token_end - 1][1]

    return (text[start:end], query_id - start, tokens[token_start:token_end], [
        i - token_start if i is not None else None
        for i in text2token[start:end]
    ], [(s - start, e - start) for s, e in token2text[token_start:token_end]])


def prepare_data(sent_path, lb_path=None):
    raw_texts = open(sent_path).read().rstrip().split('\n')
    query_ids = [raw.index(ANCHOR_CHAR) for raw in raw_texts]
    texts = [raw.replace(ANCHOR_CHAR, '') for raw in raw_texts]
    if lb_path is None:
        return texts, query_ids
    else:
        phonemes = open(lb_path).read().rstrip().split('\n')
        return texts, query_ids, phonemes


def get_phoneme_labels(polyphonic_chars):
    labels = sorted(list(set([phoneme for char, phoneme in polyphonic_chars])))
    char2phonemes = {}
    for char, phoneme in polyphonic_chars:
        if char not in char2phonemes:
            char2phonemes[char] = []
        char2phonemes[char].append(labels.index(phoneme))
    return labels, char2phonemes


def get_char_phoneme_labels(polyphonic_chars):
    labels = sorted(
        list(set([f'{char} {phoneme}' for char, phoneme in polyphonic_chars])))
    char2phonemes = {}
    for char, phoneme in polyphonic_chars:
        if char not in char2phonemes:
            char2phonemes[char] = []
        char2phonemes[char].append(labels.index(f'{char} {phoneme}'))
    return labels, char2phonemes
