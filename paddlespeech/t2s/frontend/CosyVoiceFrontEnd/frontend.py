import json
import os
import re
from functools import partial
from typing import Callable, Generator
import librosa
import inflect
import numpy as np
import onnxruntime
import paddle
import paddle
import paddle.nn.functional as F
import numpy as np
from typing import Union, Optional
import paddlespeech
from .func import fbank
def _stft(x,
         n_fft,
         n_shift,
         win_length=None,
         window="hann",
         center=True,
         pad_mode="reflect"):
    # x: [Time, Channel]
    window = window.cpu().numpy()
    x = x.cpu().numpy()
    if x.ndim == 1:
        single_channel = True
        # x: [Time] -> [Time, Channel]
        x = x[:, None]
    else:
        single_channel = False
    x = x.astype(np.float32)

    # FIXME(kamo): librosa.stft can't use multi-channel?
    # x: [Time, Channel, Freq]
    x = np.stack(
        [
            librosa.stft(
                y=x[:, ch],
                n_fft=n_fft,
                hop_length=n_shift,
                win_length=win_length,
                window=window,
                center=center,
                pad_mode=pad_mode, ).T for ch in range(x.shape[1])
        ],
        axis=1, )

    if single_channel:
        # x: [Time, Channel, Freq] -> [Time, Freq]
        x = x[:, 0]
    return x
def log_mel_spectrogram(
    audio: Union[str, np.ndarray, paddle.Tensor],
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[str] = None,
):
    """
    Compute the log-Mel spectrogram of audio using PaddlePaddle

    Parameters
    ----------
    audio: Union[str, np.ndarray, paddle.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[str]
        If given, the audio tensor is moved to this device (e.g., 'gpu:0') before STFT

    Returns
    -------
    paddle.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    N_FFT = 400  
    HOP_LENGTH = 160
    SAMPLE_RATE = 16000  
    
    if not paddle.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = paddle.to_tensor(audio.detach().cpu().numpy(), dtype='float32')
    
    if device is not None:
        if 'gpu' in device:
            place = paddle.CUDAPlace(int(device.split(':')[-1]))
        else:
            place = paddle.CPUPlace()
        audio = audio.place(place)
    if padding > 0:
        audio = F.pad(audio.unsqueeze(0), [0, padding]).squeeze(0)
    import torch
    window = paddle.to_tensor(torch.hann_window(N_FFT).cpu().numpy())

    # window = paddle.audio.functional.get_window(
    #     'hann',       
    #     N_FFT,         
    #     dtype='float32' 
    # ).cuda()
    # stft = _stft(
    #      audio, 
    #      N_FFT,
    #      HOP_LENGTH,
    #      window=window
    # )
    stft = paddle.signal.stft(
        audio, 
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=window
    )
    # stft = paddle.to_tensor(stft)
    magnitudes = stft[..., :-1].abs().square()
    if magnitudes.shape[1] > N_FFT // 2:  
        magnitudes = magnitudes[:, :N_FFT // 2 + 1, :]
    filters = mel_filters(audio.place, n_mels, N_FFT, SAMPLE_RATE)
    mel_spec = paddle.matmul(filters, magnitudes)
    log_spec = paddle.clip(mel_spec, min=1e-10).log10()
    log_spec = paddle.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.squeeze(0) 

def mel_filters(device: str, n_mels: int = 80, n_fft: int = 400, sr: int = 16000):
   
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    filters_path = "/root/paddlejob/workspace/zhangjinghong/venv_cosy/lib/python3.10/site-packages/whisper/assets/mel_filters.npz"
    with np.load(filters_path, allow_pickle=False) as f:
        return paddle.to_tensor(f[f"mel_{n_mels}"]).to(device)



try:
    import ttsfrd

    use_ttsfrd = True
except ImportError:
    print("failed to import ttsfrd, use wetext instead")
    from wetext import Normalizer as EnNormalizer
    from wetext import Normalizer as ZhNormalizer

    use_ttsfrd = False
from .file_utils import load_wav, logging
from .frontend_utils import (contains_chinese,
                                            is_only_punctuation,
                                            remove_bracket, replace_blank,
                                            replace_corner_mark,
                                            spell_out_number, split_paragraph)


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
        self.device = 'gpu:0'
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
                if paddle.device.is_compiled_with_cuda()
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
            return self._extract_text_token_generator(text), paddle.to_tensor(
                [0], dtype=paddle.int32
            ).to(self.device)
        else:
            
            text_token = self.tokenizer.encode(
                text, allowed_special=self.allowed_special
            )
            text_token = paddle.to_tensor([text_token], dtype=paddle.int32).to(self.device)
            text_token_len = paddle.to_tensor(
                [text_token.shape[1]], dtype=paddle.int32
            ).to(self.device)
            return text_token, text_token_len

    def _extract_text_token_generator(self, text_generator):
        for text in text_generator:
            text_token, _ = self._extract_text_token(text)
            for i in range(text_token.shape[1]):
                yield text_token[:, i : i + 1]

    def _extract_speech_token(self, prompt_wav):
        speech = load_wav(prompt_wav, 16000)
        
        assert (
            speech.shape[1] / 16000 <= 30
        ), "do not support extract speech token for audio longer than 30s"
        feat =log_mel_spectrogram(speech, n_mels=128)
        feat = feat.unsqueeze(0)
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
        speech_token = paddle.to_tensor([speech_token], dtype=paddle.int32).to(self.device)
        speech_token_len = paddle.to_tensor(
            [speech_token.shape[1]], dtype=paddle.int32
        ).to(self.device)
        return speech_token, speech_token_len

    def _extract_spk_embedding(self, prompt_wav):
        speech = load_wav(prompt_wav, 16000)
        speech = paddle.to_tensor(speech.detach().cpu().numpy()).cuda()
        feat = fbank(
            speech, num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat = feat - feat.mean(axis=0, keepdim=True)
        embedding = (
            self.campplus_session.run(
                None,
                {
                    self.campplus_session.get_inputs()[0]
                    .name: feat.unsqueeze(axis=0)
                    .cpu()
                    .numpy()
                },
            )[0]
            .flatten()
            .tolist()
        )
        embedding = paddle.to_tensor([embedding]).to(self.device)
        return embedding

    def _extract_speech_feat(self, prompt_wav):
        speech = load_wav(prompt_wav, 24000)
        speech_feat = (
            paddle.transpose(self.feat_extractor(speech).squeeze(axis=0),perm=[0, 1]).to(self.device)
        )

        speech_feat = speech_feat.unsqueeze(axis=0)
        speech_feat_len = paddle.to_tensor([speech_feat.shape[1]], dtype=paddle.int32).to(
            self.device
        )
        return speech_feat, speech_feat_len

    def text_normalize(self, text, split=True, text_frontend=True):
        if isinstance(text, Generator):
            logging.info("get tts_text generator, will skip text_normalize!")
            return [text]
        if "<|" in text and "|>" in text:
            text_frontend = False
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
        embedding = self.spk2info[spk_id]["embedding"]
        model_input = {
            "text": tts_text_token,
            "text_len": tts_text_token_len,
            "llm_embedding": embedding,
            "flow_embedding": embedding,
        }
        return model_input

    def frontend_zero_shot(
        self, tts_text, prompt_text, prompt_wav, resample_rate, zero_shot_spk_id
    ):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        
        if zero_shot_spk_id == "":

            prompt_text_token, prompt_text_token_len = self._extract_text_token(
                prompt_text
            )
            
            speech_feat, speech_feat_len = self._extract_speech_feat(prompt_wav)
            speech_feat=paddle.transpose(speech_feat,perm =[0,2,1])
            speech_token, speech_token_len = self._extract_speech_token(prompt_wav)
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

            embedding = self._extract_spk_embedding(prompt_wav)
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
        self, tts_text, prompt_wav, resample_rate, zero_shot_spk_id
    ):
        model_input = self.frontend_zero_shot(
            tts_text, "", prompt_wav, resample_rate, zero_shot_spk_id
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
            instruct_text
        )
        model_input["prompt_text"] = instruct_text_token
        model_input["prompt_text_len"] = instruct_text_token_len
        return model_input

    def frontend_instruct2(
        self, tts_text, instruct_text, prompt_wav, resample_rate, zero_shot_spk_id
    ):
        model_input = self.frontend_zero_shot(
            tts_text, instruct_text, prompt_wav, resample_rate, zero_shot_spk_id
        )
        del model_input["llm_prompt_speech_token"]
        del model_input["llm_prompt_speech_token_len"]
        return model_input

    def frontend_vc(self, source_speech_16k, prompt_wav, resample_rate):
        prompt_speech_token, prompt_speech_token_len = self._extract_speech_token(
            prompt_wav
        )
        prompt_speech_feat, prompt_speech_feat_len = self._extract_speech_feat(
            prompt_wav
        )
        embedding = self._extract_spk_embedding(prompt_wav)
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
