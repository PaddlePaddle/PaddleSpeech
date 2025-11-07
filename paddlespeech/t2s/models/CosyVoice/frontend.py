import json
import os
import re
from functools import partial
from typing import Callable, Generator

import inflect
import numpy as np
import onnxruntime
import paddle
import paddlespeech
import whisper
import logging
try:
    import ttsfrd

    use_ttsfrd = True
except ImportError:
    print("failed to import ttsfrd, use wetext instead")
    from wetext import Normalizer as EnNormalizer
    from wetext import Normalizer as ZhNormalizer

    use_ttsfrd = False
# split paragrah logic：
# 1. per sentence max len token_max_n, min len token_min_n, merge if last sentence len less than merge_len
# 2. cal sentence len according to lang
# 3. split sentence according to puncatation
def split_paragraph(text: str, tokenize, lang="zh", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False):
    def calc_utt_length(_text: str):
        if lang == "zh":
            return len(_text)
        else:
            return len(tokenize(_text))

    def should_merge(_text: str):
        if lang == "zh":
            return len(_text) < merge_len
        else:
            return len(tokenize(_text)) < merge_len

    if lang == "zh":
        pounc = ['。', '？', '！', '；', '：', '、', '.', '?', '!', ';']
    else:
        pounc = ['.', '?', '!', ';', ':']
    if comma_split:
        pounc.extend(['，', ','])

    if text[-1] not in pounc:
        if lang == "zh":
            text += "。"
        else:
            text += "."

    st = 0
    utts = []
    for i, c in enumerate(text):
        if c in pounc:
            if len(text[st: i]) > 0:
                utts.append(text[st: i] + c)
            if i + 1 < len(text) and text[i + 1] in ['"', '”']:
                tmp = utts.pop(-1)
                utts.append(tmp + text[i + 1])
                st = i + 2
            else:
                st = i + 1

    final_utts = []
    cur_utt = ""
    for utt in utts:
        if calc_utt_length(cur_utt + utt) > token_max_n and calc_utt_length(cur_utt) > token_min_n:
            final_utts.append(cur_utt)
            cur_utt = ""
        cur_utt = cur_utt + utt
    if len(cur_utt) > 0:
        if should_merge(cur_utt) and len(final_utts) != 0:
            final_utts[-1] = final_utts[-1] + cur_utt
        else:
            final_utts.append(cur_utt)

    return final_utts

# spell Arabic numerals
def spell_out_number(text: str, inflect_parser):
    new_text = []
    st = None
    for i, c in enumerate(text):
        if not c.isdigit():
            if st is not None:
                num_str = inflect_parser.number_to_words(text[st: i])
                new_text.append(num_str)
                st = None
            new_text.append(c)
        else:
            if st is None:
                st = i
    if st is not None and st < len(text):
        num_str = inflect_parser.number_to_words(text[st:])
        new_text.append(num_str)
    return ''.join(new_text)

# replace special symbol
def replace_corner_mark(text):
    text = text.replace('²', '平方')
    text = text.replace('³', '立方')
    return text

# remove blank between chinese character
def replace_blank(text: str):
    out_str = []
    for i, c in enumerate(text):
        if c == " ":
            if ((text[i + 1].isascii() and text[i + 1] != " ") and
                    (text[i - 1].isascii() and text[i - 1] != " ")):
                out_str.append(c)
        else:
            out_str.append(c)
    return "".join(out_str)
def is_only_punctuation(text):
    # Regular expression: Match strings that consist only of punctuation marks or are empty.
    punctuation_pattern = r'^[\p{P}\p{S}]*$'
    return bool(regex.fullmatch(punctuation_pattern, text))

# remove meaningless symbol
def remove_bracket(text):
    text = text.replace('（', '').replace('）', '')
    text = text.replace('【', '').replace('】', '')
    text = text.replace('`', '').replace('`', '')
    text = text.replace("——", " ")
    return text
class CosyVoiceFrontEnd:
    def __init__(
        self,
        get_tokenizer: Callable,
        feat_extractor: Callable,
        campplus_model: str,
        speech_tokenizer_model: str,
        spk2info: str = "",
        allowed_special: str = "all",
    ):
        self.tokenizer = get_tokenizer()
        self.feat_extractor = feat_extractor
        self.device = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        option.intra_op_num_threads = 1
        self.campplus_session = onnxruntime.InferenceSession(
            campplus_model, sess_options=option, providers=["CPUExecutionProvider"]
        )
        self.speech_tokenizer_session = onnxruntime.InferenceSession(
            speech_tokenizer_model,
            sess_options=option,
            providers=[
                "CUDAExecutionProvider"
                if paddle.device.cuda.device_count() >= 1
                else "CPUExecutionProvider"
            ],
        )
        if os.path.exists(spk2info):
            self.spk2info = paddle.load(path=str(spk2info))
        else:
            self.spk2info = {}
        self.allowed_special = allowed_special
        self.use_ttsfrd = use_ttsfrd
        if self.use_ttsfrd:
            self.frd = ttsfrd.TtsFrontendEngine()
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            assert (
                self.frd.initialize(
                    "{}/../../pretrained_models/CosyVoice-ttsfrd/resource".format(
                        ROOT_DIR
                    )
                )
                is True
            ), "failed to initialize ttsfrd resource"
            self.frd.set_lang_type("pinyinvg")
        else:
            self.zh_tn_model = ZhNormalizer(remove_erhua=False)
            self.en_tn_model = EnNormalizer()
            self.inflect_parser = inflect.engine()

    def _extract_text_token(self, text):
        if isinstance(text, Generator):
            logging.info(
                "get tts_text generator, will return _extract_text_token_generator!"
            )
            return self._extract_text_token_generator(text), paddle.tensor(
                [0], dtype=paddle.int32
            ).to(self.device)
        else:
            text_token = self.tokenizer.encode(
                text, allowed_special=self.allowed_special
            )
            text_token = paddle.tensor([text_token], dtype=paddle.int32).to(self.device)
            text_token_len = paddle.tensor(
                [text_token.shape[1]], dtype=paddle.int32
            ).to(self.device)
            return text_token, text_token_len

    def _extract_text_token_generator(self, text_generator):
        for text in text_generator:
            text_token, _ = self._extract_text_token(text)
            for i in range(text_token.shape[1]):
                yield text_token[:, i : i + 1]

    def _extract_speech_token(self, speech):
        assert (
            speech.shape[1] / 16000 <= 30
        ), "do not support extract speech token for audio longer than 30s"
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_token = (
            self.speech_tokenizer_session.run(
                None,
                {
                    self.speech_tokenizer_session.get_inputs()[0]
                    .name: feat.detach()
                    .cpu()
                    .numpy(),
                    self.speech_tokenizer_session.get_inputs()[1].name: np.array(
                        [feat.shape[2]], dtype=np.int32
                    ),
                },
            )[0]
            .flatten()
            .tolist()
        )
        speech_token = paddle.tensor([speech_token], dtype=paddle.int32).to(self.device)
        speech_token_len = paddle.tensor(
            [speech_token.shape[1]], dtype=paddle.int32
        ).to(self.device)
        return speech_token, speech_token_len

    def _extract_spk_embedding(self, speech):
        ##################>>>>>>>>>>>>>>>>>>> 
        feat = torchaudio.compliance.kaldi.fbank(
            speech, num_mel_bins=80, dither=0, sample_frequency=16000
        )
        ##################>>>>>>>>>>>>>>>>>>> 
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = (
            self.campplus_session.run(
                None,
                {
                    self.campplus_session.get_inputs()[0]
                    .name: feat.unsqueeze(dim=0)
                    .cpu()
                    .numpy()
                },
            )[0]
            .flatten()
            .tolist()
        )
        embedding = paddle.tensor([embedding]).to(self.device)
        return embedding

    def _extract_speech_feat(self, speech):
        speech_feat = (
            self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        )
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = paddle.tensor([speech_feat.shape[1]], dtype=paddle.int32).to(
            self.device
        )
        return speech_feat, speech_feat_len

    def text_normalize(self, text, split=True, text_frontend=True):
        if isinstance(text, Generator):
            logging.info("get tts_text generator, will skip text_normalize!")
            return [text]
        if text_frontend is False or text == "":
            return [text] if split is True else text
        text = text.strip()
        if self.use_ttsfrd:
            texts = [
                i["text"]
                for i in json.loads(self.frd.do_voicegen_frd(text))["sentences"]
            ]
            text = "".join(texts)
        elif contains_chinese(text):
            text = self.zh_tn_model.normalize(text)
            text = text.replace("\n", "")
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace(".", "。")
            text = text.replace(" - ", "，")
            text = remove_bracket(text)
            text = re.sub("[，,、]+$", "。", text)
            texts = list(
                split_paragraph(
                    text,
                    partial(
                        self.tokenizer.encode, allowed_special=self.allowed_special
                    ),
                    "zh",
                    token_max_n=80,
                    token_min_n=60,
                    merge_len=20,
                    comma_split=False,
                )
            )
        else:
            text = self.en_tn_model.normalize(text)
            text = spell_out_number(text, self.inflect_parser)
            texts = list(
                split_paragraph(
                    text,
                    partial(
                        self.tokenizer.encode, allowed_special=self.allowed_special
                    ),
                    "en",
                    token_max_n=80,
                    token_min_n=60,
                    merge_len=20,
                    comma_split=False,
                )
            )
        texts = [i for i in texts if not is_only_punctuation(i)]
        return texts if split is True else text

    def frontend_sft(self, tts_text, spk_id):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        print("1" * 30)
        print(self.spk2info.keys())
        embedding = self.spk2info[spk_id]["embedding"]
        model_input = {
            "text": tts_text_token,
            "text_len": tts_text_token_len,
            "llm_embedding": embedding,
            "flow_embedding": embedding,
        }
        return model_input

    def frontend_zero_shot(
        self, tts_text, prompt_text, prompt_speech_16k, resample_rate, zero_shot_spk_id
    ):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        if zero_shot_spk_id == "":
            prompt_text_token, prompt_text_token_len = self._extract_text_token(
                prompt_text
            )
            #>>>>>>>>>>>>>>>>>>> 
            prompt_speech_resample = torchaudio.transforms.Resample(
                orig_freq=16000, new_freq=resample_rate
            )(prompt_speech_16k)
            #>>>>>>>>>>>>>>>>>>>
            speech_feat, speech_feat_len = self._extract_speech_feat(
                prompt_speech_resample
            )
            speech_token, speech_token_len = self._extract_speech_token(
                prompt_speech_16k
            )
            if resample_rate == 24000:
                token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
                speech_feat, speech_feat_len[:] = (
                    speech_feat[:, : 2 * token_len],
                    2 * token_len,
                )
                speech_token, speech_token_len[:] = (
                    speech_token[:, :token_len],
                    token_len,
                )
            embedding = self._extract_spk_embedding(prompt_speech_16k)
            model_input = {
                "prompt_text": prompt_text_token,
                "prompt_text_len": prompt_text_token_len,
                "llm_prompt_speech_token": speech_token,
                "llm_prompt_speech_token_len": speech_token_len,
                "flow_prompt_speech_token": speech_token,
                "flow_prompt_speech_token_len": speech_token_len,
                "prompt_speech_feat": speech_feat,
                "prompt_speech_feat_len": speech_feat_len,
                "llm_embedding": embedding,
                "flow_embedding": embedding,
            }
        else:
            model_input = self.spk2info[zero_shot_spk_id]
        model_input["text"] = tts_text_token
        model_input["text_len"] = tts_text_token_len
        return model_input

    def frontend_cross_lingual(
        self, tts_text, prompt_speech_16k, resample_rate, zero_shot_spk_id
    ):
        model_input = self.frontend_zero_shot(
            tts_text, "", prompt_speech_16k, resample_rate, zero_shot_spk_id
        )
        del model_input["prompt_text"]
        del model_input["prompt_text_len"]
        del model_input["llm_prompt_speech_token"]
        del model_input["llm_prompt_speech_token_len"]
        return model_input

    def frontend_instruct(self, tts_text, spk_id, instruct_text):
        model_input = self.frontend_sft(tts_text, spk_id)
        del model_input["llm_embedding"]
        instruct_text_token, instruct_text_token_len = self._extract_text_token(
            instruct_text + "<endofprompt>"
        )
        model_input["prompt_text"] = instruct_text_token
        model_input["prompt_text_len"] = instruct_text_token_len
        return model_input

    def frontend_instruct2(
        self,
        tts_text,
        instruct_text,
        prompt_speech_16k,
        resample_rate,
        zero_shot_spk_id,
    ):
        model_input = self.frontend_zero_shot(
            tts_text,
            instruct_text + "<|endofprompt|>",
            prompt_speech_16k,
            resample_rate,
            zero_shot_spk_id,
        )
        del model_input["llm_prompt_speech_token"]
        del model_input["llm_prompt_speech_token_len"]
        return model_input

    def frontend_vc(self, source_speech_16k, prompt_speech_16k, resample_rate):
        prompt_speech_token, prompt_speech_token_len = self._extract_speech_token(
            prompt_speech_16k
        )
        #>>>>>>>>>>>>>>>>>>  
        prompt_speech_resample = torchaudio.transforms.Resample(
            orig_freq=16000, new_freq=resample_rate
        )(prompt_speech_16k)
        #>>>>>>>>>>>>>>>>>>
        prompt_speech_feat, prompt_speech_feat_len = self._extract_speech_feat(
            prompt_speech_resample
        )
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        source_speech_token, source_speech_token_len = self._extract_speech_token(
            source_speech_16k
        )
        model_input = {
            "source_speech_token": source_speech_token,
            "source_speech_token_len": source_speech_token_len,
            "flow_prompt_speech_token": prompt_speech_token,
            "flow_prompt_speech_token_len": prompt_speech_token_len,
            "prompt_speech_feat": prompt_speech_feat,
            "prompt_speech_feat_len": prompt_speech_feat_len,
            "flow_embedding": embedding,
        }
        return model_input