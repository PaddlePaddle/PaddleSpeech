"""
Credits
 This code is modified from https://github.com/GitYCC/g2pW
"""
import os
import json
import onnxruntime
import numpy as np

from opencc import OpenCC

from paddlenlp.transformers import BertTokenizer
from paddlespeech.utils.env import MODEL_HOME
from paddlespeech.t2s.frontend.g2pw.dataset import prepare_data,\
                                                   prepare_onnx_input,\
                                                   get_phoneme_labels,\
                                                   get_char_phoneme_labels
from paddlespeech.t2s.frontend.g2pw.utils import load_config
from paddlespeech.cli.utils import download_and_decompress
from paddlespeech.resource.pretrained_models import g2pw_onnx_models


def predict(session, onnx_input, labels):
    all_preds = []
    all_confidences = []
    probs = session.run([],{"input_ids": onnx_input['input_ids'],
                            "token_type_ids":onnx_input['token_type_ids'],
                            "attention_mask":onnx_input['attention_masks'],
                            "phoneme_mask":onnx_input['phoneme_masks'],
                            "char_ids":onnx_input['char_ids'],
                            "position_ids":onnx_input['position_ids']})[0]
                            
    preds = np.argmax(probs,axis=1).tolist()
    max_probs = []
    for index,arr in zip(preds,probs.tolist()):
        max_probs.append(arr[index])
    all_preds += [labels[pred] for pred in preds]
    all_confidences += max_probs

    return all_preds, all_confidences


class G2PWOnnxConverter:
    def __init__(self, model_dir = MODEL_HOME, style='bopomofo', model_source=None, enable_non_tradional_chinese=False):
        if not os.path.exists(os.path.join(model_dir, 'G2PWModel/g2pW.onnx')):
            uncompress_path = download_and_decompress(g2pw_onnx_models['G2PWModel']['1.0'],model_dir)

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = 2
        self.session_g2pW =  onnxruntime.InferenceSession(os.path.join(model_dir, 'G2PWModel/g2pW.onnx'),sess_options=sess_options)
        self.config = load_config(os.path.join(model_dir, 'G2PWModel/config.py'), use_default=True)

        self.model_source = model_source if model_source else self.config.model_source
        self.enable_opencc = enable_non_tradional_chinese

        self.tokenizer = BertTokenizer.from_pretrained(self.config.model_source)

        polyphonic_chars_path = os.path.join(model_dir, 'G2PWModel/POLYPHONIC_CHARS.txt')
        monophonic_chars_path = os.path.join(model_dir, 'G2PWModel/MONOPHONIC_CHARS.txt')
        self.polyphonic_chars = [line.split('\t') for line in open(polyphonic_chars_path,encoding='utf-8').read().strip().split('\n')]
        self.monophonic_chars = [line.split('\t') for line in open(monophonic_chars_path,encoding='utf-8').read().strip().split('\n')]
        self.labels, self.char2phonemes = get_char_phoneme_labels(self.polyphonic_chars) if self.config.use_char_phoneme else get_phoneme_labels(self.polyphonic_chars)

        self.chars = sorted(list(self.char2phonemes.keys()))
        self.pos_tags = ['UNK', 'A', 'C', 'D', 'I', 'N', 'P', 'T', 'V', 'DE', 'SHI']

        with open(os.path.join(model_dir,'G2PWModel/bopomofo_to_pinyin_wo_tune_dict.json'), 'r',encoding='utf-8') as fr:
            self.bopomofo_convert_dict = json.load(fr)
        self.style_convert_func = {
            'bopomofo': lambda x: x,
            'pinyin': self._convert_bopomofo_to_pinyin,
        }[style]

        with open(os.path.join(model_dir,'G2PWModel/char_bopomofo_dict.json'), 'r',encoding='utf-8') as fr:
            self.char_bopomofo_dict = json.load(fr)

        if self.enable_opencc:
            self.cc = OpenCC('s2tw')

    def _convert_bopomofo_to_pinyin(self, bopomofo):
        tone = bopomofo[-1]
        assert tone in '12345'
        component = self.bopomofo_convert_dict.get(bopomofo[:-1])
        if component:
            return component + tone
        else:
            print(f'Warning: "{bopomofo}" cannot convert to pinyin')
            return None

    def __call__(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]

        if self.enable_opencc:
            translated_sentences = []
            for sent in sentences:
                translated_sent = self.cc.convert(sent)
                assert len(translated_sent) == len(sent)
                translated_sentences.append(translated_sent)
            sentences = translated_sentences
        
        texts, query_ids, sent_ids, partial_results = self._prepare_data(sentences)
        if len(texts) == 0:
            # sentences no polyphonic words
            return partial_results

        onnx_input = prepare_onnx_input(self.tokenizer, self.labels, self.char2phonemes, self.chars, texts, query_ids,
                              use_mask=self.config.use_mask, use_char_phoneme=self.config.use_char_phoneme,
                              window_size=self.config.window_size)

        preds, confidences = predict(self.session_g2pW, onnx_input, self.labels)
        if self.config.use_char_phoneme:
            preds = [pred.split(' ')[1] for pred in preds]

        results = partial_results
        for sent_id, query_id, pred in zip(sent_ids, query_ids, preds):
            results[sent_id][query_id] = self.style_convert_func(pred)

        return results

    def _prepare_data(self, sentences):
        polyphonic_chars = set(self.chars)
        monophonic_chars_dict = {
            char: phoneme for char, phoneme in self.monophonic_chars
        }
        texts, query_ids, sent_ids, partial_results = [], [], [], []
        for sent_id, sent in enumerate(sentences):
            partial_result = [None] * len(sent)
            for i, char in enumerate(sent):
                if char in polyphonic_chars:
                    texts.append(sent)
                    query_ids.append(i)
                    sent_ids.append(sent_id)
                elif char in monophonic_chars_dict:
                    partial_result[i] =  self.style_convert_func(monophonic_chars_dict[char])
                elif char in self.char_bopomofo_dict:
                    partial_result[i] =  self.style_convert_func(self.char_bopomofo_dict[char][0])
            partial_results.append(partial_result)
        return texts, query_ids, sent_ids, partial_results
