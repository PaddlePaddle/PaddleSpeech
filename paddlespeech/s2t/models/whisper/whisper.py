# MIT License, Copyright (c) 2022 OpenAI.
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Modified from OpenAI Whisper 2022 (https://github.com/openai/whisper/whisper)
import os
from dataclasses import dataclass
from dataclasses import field
from functools import lru_cache
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import paddle
import paddle.nn.functional as F
import soundfile
import tqdm
from paddle import nn
from paddle.distribution import Categorical

import paddlespeech.s2t.modules.align as paddlespeech_nn
from paddlespeech.s2t.models.whisper import utils
from paddlespeech.s2t.models.whisper.tokenizer import get_tokenizer
from paddlespeech.s2t.models.whisper.tokenizer import LANGUAGES
from paddlespeech.s2t.models.whisper.tokenizer import Tokenizer
from paddlespeech.s2t.utils.log import Log
logger = Log(__name__).getlog()

_MODELS = ["large"]
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000: number of samples in a chunk
N_FRAMES = utils.exact_div(
    N_SAMPLES, HOP_LENGTH)  # 3000: number of frames in a mel spectrogram input


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(paddlespeech_nn.LayerNorm):
    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return super().forward(x)


class Linear(paddlespeech_nn.Linear):
    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return F.linear(x, self.weight, None
                        if self.bias is None else self.bias)


class Conv1d(paddlespeech_nn.Conv1D):
    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return super().forward(x)


class MultiHeadAttention(nn.Layer):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state, bias_attr=True)
        self.key = Linear(n_state, n_state, bias_attr=False)
        self.value = Linear(n_state, n_state, bias_attr=True)
        self.out = Linear(n_state, n_state, bias_attr=True)

    def forward(
            self,
            x: paddle.Tensor,
            xa: Optional[paddle.Tensor]=None,
            mask: Optional[paddle.Tensor]=None,
            kv_cache: Optional[dict]=None, ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv = self.qkv_attention(q, k, v, mask)
        return self.out(wv)

    def qkv_attention(self,
                      q: paddle.Tensor,
                      k: paddle.Tensor,
                      v: paddle.Tensor,
                      mask: Optional[paddle.Tensor]=None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head)**-0.25
        q = paddle.transpose(
            q.view(*q.shape[:2], self.n_head, -1), (0, 2, 1, 3)) * scale
        k = paddle.transpose(
            k.view(*k.shape[:2], self.n_head, -1), (0, 2, 3, 1)) * scale
        v = paddle.transpose(
            v.view(*v.shape[:2], self.n_head, -1), (0, 2, 1, 3))

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]

        w = F.softmax(qk.float(), axis=-1).to(q.dtype)
        return paddle.transpose((w @ v), (0, 2, 1, 3)).flatten(start_axis=2)


class ResidualAttentionBlock(nn.Layer):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool=False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(
            n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp, bias_attr=True),
            nn.GELU(), Linear(n_mlp, n_state, bias_attr=True))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
            self,
            x: paddle.Tensor,
            xa: Optional[paddle.Tensor]=None,
            mask: Optional[paddle.Tensor]=None,
            kv_cache: Optional[dict]=None, ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
        if self.cross_attn:
            x = x + self.cross_attn(
                self.cross_attn_ln(x), xa, kv_cache=kv_cache)
        x = x + self.mlp(self.mlp_ln(x))
        return x


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = paddle.exp(-log_timescale_increment * paddle.arange(
        channels // 2, dtype=paddle.float32))
    scaled_time = paddle.arange(
        length,
        dtype=paddle.float32)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return paddle.to_tensor(
        paddle.concat(
            [paddle.sin(scaled_time), paddle.cos(scaled_time)], axis=1))


class AudioEncoder(nn.Layer):
    def __init__(self,
                 n_mels: int,
                 n_ctx: int,
                 n_state: int,
                 n_head: int,
                 n_layer: int):
        super().__init__()
        self.conv1 = Conv1d(
            n_mels, n_state, kernel_size=3, stride=1, padding=1, bias_attr=True)
        self.conv2 = Conv1d(
            n_state,
            n_state,
            kernel_size=3,
            stride=2,
            padding=1,
            bias_attr=True)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.LayerList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)])
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: paddle.Tensor):
        """
        x : paddle.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = paddle.transpose(x, (0, 2, 1))

        assert x.shape[
            1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Layer):
    def __init__(self,
                 n_vocab: int,
                 n_ctx: int,
                 n_state: int,
                 n_head: int,
                 n_layer: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = paddle.create_parameter(
            shape=[n_ctx, n_state], dtype='float32')

        self.blocks: Iterable[ResidualAttentionBlock] = nn.LayerList([
            ResidualAttentionBlock(n_state, n_head, cross_attention=True)
            for _ in range(n_layer)
        ])
        self.ln = LayerNorm(n_state)

        mask = paddle.full(
            shape=[n_ctx, n_state], fill_value=-np.inf, dtype='float32')
        mask = paddle.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistable=False)

    def forward(self,
                x: paddle.Tensor,
                xa: paddle.Tensor,
                kv_cache: Optional[dict]=None):
        """
        x : paddle.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : paddle.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = self.token_embedding(x) + self.positional_embedding[offset:offset +
                                                                x.shape[-1]]
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (x @ paddle.transpose(self.token_embedding.weight, (1, 0)))

        return logits


@dataclass(frozen=True)
class DecodingOptions:
    task: str = "transcribe"  # whether to perform X->X "transcribe" or X->English "translate"
    language: Optional[
        str] = None  # language that the audio is in; uses detected language if None
    # sampling-related options
    temperature: float = 0.0
    sample_len: Optional[int] = None  # maximum number of tokens to sample
    best_of: Optional[
        int] = None  # number of independent samples to collect, when t > 0
    beam_size: Optional[
        int] = None  # number of beams in beam search, when t == 0
    patience: Optional[
        float] = None  # patience in beam search (https://arxiv.org/abs/2204.05424)

    # options for ranking generations (either beams or best-of-N samples)
    length_penalty: Optional[
        float] = None  # "alpha" in Google NMT, None defaults to length norm

    # prompt, prefix, and token suppression
    prompt: Optional[Union[str, List[
        int]]] = None  # text or tokens for the previous context
    prefix: Optional[Union[str, List[
        int]]] = None  # text or tokens to prefix the current context
    suppress_blank: bool = True  # this will suppress blank outputs

    # list of tokens ids (or comma-separated token ids) to suppress
    # "-1" will suppress a set of symbols as defined in `tokenizer.non_speech_tokens()`
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"

    # timestamp sampling options
    without_timestamps: bool = False  # use <|notimestamps|> to sample text tokens only
    max_initial_timestamp: Optional[
        float] = 1.0  # the initial timestamp cannot be later than this

    # implementation details
    fp16: bool = False  # use fp16 for most of the calculation


@dataclass(frozen=True)
class DecodingResult:
    audio_features: paddle.Tensor
    language: str
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan


class Inference:
    def logits(self, tokens: paddle.Tensor,
               audio_features: paddle.Tensor) -> paddle.Tensor:
        """Perform a forward pass on the decoder and return per-token logits"""
        raise NotImplementedError

    def rearrange_kv_cache(self, source_indices) -> None:
        """Update the key-value cache according to the updated beams"""
        raise NotImplementedError

    def cleanup_caching(self) -> None:
        """Clean up any resources or hooks after decoding is finished"""
        pass


class WhisperInference(Inference):
    def __init__(self, model: "Whisper", initial_token_length: int):
        self.model: "Whisper" = model
        self.initial_token_length = initial_token_length
        self.kv_cache = {}
        self.hooks = []

    def logits(self, tokens: paddle.Tensor,
               audio_features: paddle.Tensor) -> paddle.Tensor:
        if not self.kv_cache:
            self.kv_cache, self.hooks = self.model.install_kv_cache_hooks()

        if tokens.shape[-1] > self.initial_token_length:
            # only need to use the last token except in the first forward pass
            tokens = tokens[:, -1:]

        return self.model.decoder(
            tokens, audio_features, kv_cache=self.kv_cache)

    def cleanup_caching(self):
        for hook in self.hooks:
            hook.remove()

        self.kv_cache = {}
        self.hooks = []

    def rearrange_kv_cache(self, source_indices):
        for module, tensor in self.kv_cache.items():
            # update the key/value cache to contain the selected sequences
            self.kv_cache[module] = tensor[source_indices].detach()


@paddle.no_grad()
def detect_language(
        model: "Whisper",
        mel: paddle.Tensor,
        resource_path: str,
        tokenizer: Tokenizer=None) -> Tuple[paddle.Tensor, List[dict]]:
    """
    Detect the spoken language in the audio, and return them as list of strings, along with the ids
    of the most probable language tokens and the probability distribution over all language tokens.
    This is performed outside the main decode loop in order to not interfere with kv-caching.

    Returns
    -------
    language_tokens : Tensor, shape = (batch_size,)
        ids of the most probable language tokens, which appears after the startoftranscript token.
    language_probs : List[Dict[str, float]], length = batch_size
        list of dictionaries containing the probability distribution over all languages.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(
            model.is_multilingual, resource_path=resource_path)
    if tokenizer.language is None or tokenizer.language_token not in tokenizer.sot_sequence:
        raise ValueError(
            "This model doesn't have language tokens so it can't perform lang id"
        )

    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)

    # skip encoder forward pass if already-encoded audio features were given
    if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state):
        mel = model.encoder(mel)

    # forward pass using a single token, startoftranscript
    batch_size = mel.shape[0]
    x = paddle.to_tensor([[tokenizer.sot]] * batch_size)  # [batch_size, 1]
    logits = model.logits(x, mel)[:, 0]

    # collect detected languages; suppress all non-language tokens
    mask = paddle.ones(paddle.to_tensor(logits.shape[-1]), dtype=bool)
    mask[list(tokenizer.all_language_tokens)] = False
    logits[:, mask] = -np.inf
    language_tokens = paddle.argmax(logits, axis=-1)
    language_token_probs = F.softmax(logits, axis=-1)
    language_probs = [{
        c: language_token_probs[i, j].tolist()
        for j, c in zip(tokenizer.all_language_tokens,
                        tokenizer.all_language_codes)
    } for i in range(batch_size)]

    if single:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]

    return language_tokens, language_probs


def transcribe(
        model: "Whisper",
        mel: paddle.Tensor,
        resource_path: str,
        *,
        verbose: Optional[bool]=None,
        temperature: Union[float, Tuple[float, ...]]=(0.0, 0.2, 0.4, 0.6, 0.8,
                                                      1.0),
        compression_ratio_threshold: Optional[float]=2.4,
        logprob_threshold: Optional[float]=-1.0,
        no_speech_threshold: Optional[float]=0.6,
        condition_on_previous_text: bool=True,
        **decode_options, ):
    """
    Transcribe an audio file using Whisper

    Parameters
    ----------
    model: Whisper
        The Whisper model instance

    mel: paddle.Tensor
        The audio feature

    verbose: bool
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything

    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successfully used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed

    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed

    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent

    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances

    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """
    dtype = np.float32  #paddle only support float32

    if dtype == np.float32:
        decode_options["fp16"] = False

    if decode_options.get("language") == 'None' or decode_options.get(
            "language", None) is None:
        if not model.is_multilingual:
            decode_options["language"] = "en"
        else:
            if verbose:
                print(
                    "Detecting language using up to the first 30 seconds. Use `--language` to specify the language"
                )
            segment = pad_or_trim(mel, N_FRAMES)
            _, probs = model.detect_language(segment, resource_path)
            decode_options["language"] = max(probs, key=probs.get)
            if verbose is not None:
                print(
                    f"Detected language: {LANGUAGES[decode_options['language']].title()}"
                )

    language = decode_options["language"]
    task = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(
        model.is_multilingual,
        resource_path=resource_path,
        language=language,
        task=task)

    def decode_with_fallback(segment: paddle.Tensor) -> DecodingResult:
        temperatures = [temperature] if isinstance(temperature, (
            int, float)) else temperature
        decode_result = None

        for t in temperatures:
            kwargs = {**decode_options}
            if t > 0:
                # disable beam_size and patience when t > 0
                kwargs.pop("beam_size", None)
                kwargs.pop("patience", None)
            else:
                # disable best_of when t == 0
                kwargs.pop("best_of", None)

            options = DecodingOptions(**kwargs, temperature=t)
            decode_result = model.decode(segment, options, resource_path)

            needs_fallback = False
            if compression_ratio_threshold is not None and decode_result.compression_ratio > compression_ratio_threshold:
                needs_fallback = True  # too repetitive
            if logprob_threshold is not None and decode_result.avg_logprob < logprob_threshold:
                needs_fallback = True  # average log probability is too low

            if not needs_fallback:
                break

        return decode_result

    seek = 0
    input_stride = utils.exact_div(
        N_FRAMES, model.dims.n_audio_ctx)  # mel frames per output token: 2
    time_precision = (input_stride * HOP_LENGTH /
                      SAMPLE_RATE)  # time per output token: 0.02 (seconds)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    initial_prompt = decode_options.pop("initial_prompt", None) or []
    if initial_prompt:
        initial_prompt = tokenizer.encode(" " +
                                          initial_prompt.strip()).input_ids
        all_tokens.extend(initial_prompt)

    def add_segment(*,
                    start: float,
                    end: float,
                    text_tokens: paddle.Tensor,
                    result: DecodingResult):
        text = tokenizer.decode(
            [token for token in text_tokens if token < tokenizer.eot])
        if len(text.strip()) == 0:  # skip empty text output
            return

        all_segments.append({
            "id": len(all_segments),
            "seek": seek,
            "start": start,
            "end": end,
            "text": text,
            "tokens": result.tokens,
            "temperature": result.temperature,
            "avg_logprob": result.avg_logprob,
            "compression_ratio": result.compression_ratio,
            "no_speech_prob": result.no_speech_prob,
        })
        if verbose:
            print(
                f"[{utils.format_timestamp(start)} --> {utils.format_timestamp(end)}] {text}"
            )

    # show the progress bar when verbose is False (otherwise the transcribed text will be printed)
    num_frames = mel.shape[-1]
    previous_seek_value = seek

    with tqdm.tqdm(
            total=num_frames, unit='frames',
            disable=verbose is not False) as pbar:
        while seek < num_frames:
            timestamp_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
            segment = pad_or_trim(mel[:, seek:], N_FRAMES)
            segment_duration = segment.shape[-1] * HOP_LENGTH / SAMPLE_RATE

            decode_options["prompt"] = all_tokens[prompt_reset_since:]
            result: DecodingResult = decode_with_fallback(segment)
            tokens = paddle.to_tensor(result.tokens)

            if no_speech_threshold is not None:
                # no voice activity check
                should_skip = result.no_speech_prob > no_speech_threshold
                if logprob_threshold is not None and result.avg_logprob > logprob_threshold:
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    should_skip = False

                if should_skip:
                    seek += segment.shape[
                        -1]  # fast-forward to the next segment boundary
                    continue

            timestamp_tokens: paddle.Tensor = tokens.greater_equal(
                paddle.to_tensor(tokenizer.timestamp_begin))

            consecutive = paddle.where(timestamp_tokens[:-1] & timestamp_tokens[
                1:])[0]
            if len(
                    consecutive
            ) > 0:  # if the output contains two consecutive timestamp tokens
                consecutive = paddle.add(consecutive, paddle.to_tensor(1))
                last_slice = 0
                for current_slice in consecutive:
                    sliced_tokens = tokens[last_slice:current_slice]
                    start_timestamp_position = (
                        sliced_tokens[0].item() - tokenizer.timestamp_begin)
                    end_timestamp_position = (
                        sliced_tokens[-1].item() - tokenizer.timestamp_begin)
                    add_segment(
                        start=timestamp_offset + start_timestamp_position *
                        time_precision,
                        end=timestamp_offset + end_timestamp_position *
                        time_precision,
                        text_tokens=sliced_tokens[1:-1],
                        result=result, )
                    last_slice = current_slice
                last_timestamp_position = (
                    tokens[last_slice - 1].item() - tokenizer.timestamp_begin)
                seek += last_timestamp_position * input_stride
                all_tokens.extend(tokens[:last_slice + 1].tolist())
            else:
                duration = segment_duration
                timestamps = tokens[timestamp_tokens.nonzero().flatten()]
                if len(timestamps) > 0 and timestamps[
                        -1].item() != tokenizer.timestamp_begin:
                    # no consecutive timestamps but it has a timestamp; use the last one.
                    # single timestamp at the end means no speech after the last timestamp.
                    last_timestamp_position = timestamps[
                        -1].item() - tokenizer.timestamp_begin
                    duration = last_timestamp_position * time_precision

                add_segment(
                    start=timestamp_offset,
                    end=timestamp_offset + duration,
                    text_tokens=tokens,
                    result=result, )

                seek += segment.shape[-1]
                all_tokens.extend(tokens.tolist())

            if not condition_on_previous_text or result.temperature > 0.5:
                # do not feed the prompt tokens if a high temperature was used
                prompt_reset_since = len(all_tokens)

            # update progress bar
            pbar.update(min(num_frames, seek) - previous_seek_value)
            previous_seek_value = seek

    return dict(
        text=tokenizer.decode(all_tokens[len(initial_prompt):]),
        segments=all_segments,
        language=language)


class SequenceRanker:
    def rank(self,
             tokens: List[List[paddle.Tensor]],
             sum_logprobs: List[List[float]]) -> List[int]:
        """
        Given a list of groups of samples and their cumulative log probabilities,
        return the indices of the samples in each group to select as the final result
        """
        raise NotImplementedError


class MaximumLikelihoodRanker(SequenceRanker):
    """
    Select the sample with the highest log probabilities, penalized using either
    a simple length normalization or Google NMT paper's length penalty
    """

    def __init__(self, length_penalty: Optional[float]):
        self.length_penalty = length_penalty

    def rank(self,
             tokens: List[List[paddle.Tensor]],
             sum_logprobs: List[List[float]]):
        def scores(logprobs, lengths):
            result = []
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty is None:
                    penalty = length
                else:
                    # from the Google NMT paper
                    penalty = ((5 + length) / 6)**self.length_penalty
                result.append(logprob / penalty)
            return result

        # get the sequence with the highest score
        lengths = [[len(t) for t in s] for s in tokens]
        return [np.argmax(scores(p, l)) for p, l in zip(sum_logprobs, lengths)]


class TokenDecoder:
    def reset(self):
        """Initialize any stateful variables for decoding a new sequence"""

    def update(self,
               tokens: paddle.Tensor,
               logits: paddle.Tensor,
               sum_logprobs: paddle.Tensor) -> Tuple[paddle.Tensor, bool]:
        """Specify how to select the next token, based on the current trace and logits

        Parameters
        ----------
        tokens : Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        logits : Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        sum_logprobs : Tensor, shape = (n_batch)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Tensor, shape = (n_batch, current_sequence_length + 1)
            the tokens, appended with the selected next token

        completed : bool
            True if all sequences has reached the end of text

        """
        raise NotImplementedError

    def finalize(
            self, tokens: paddle.Tensor, sum_logprobs: paddle.Tensor
    ) -> Tuple[Sequence[Sequence[paddle.Tensor]], List[List[float]]]:
        """Finalize search and return the final candidate sequences

        Parameters
        ----------
        tokens : Tensor, shape = (batch_size, beam_size, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence

        sum_logprobs : Tensor, shape = (batch_size, beam_size)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Sequence[Sequence[Tensor]], length = batch_size
            sequence of Tensors containing candidate token sequences, for each audio input

        sum_logprobs : List[List[float]], length = batch_size
            sequence of cumulative log probabilities corresponding to the above

        """
        raise NotImplementedError


class GreedyDecoder(TokenDecoder):
    def __init__(self, temperature: float, eot: int):
        self.temperature = temperature
        self.eot = eot

    def update(self,
               tokens: paddle.Tensor,
               logits: paddle.Tensor,
               sum_logprobs: paddle.Tensor) -> Tuple[paddle.Tensor, bool]:
        temperature = self.temperature
        if temperature == 0:
            next_tokens = paddle.argmax(logits, axis=-1)
        else:
            next_tokens = Categorical(logits=logits / temperature).sample([1])
            next_tokens = paddle.reshape(next_tokens, [
                next_tokens.shape[0] * next_tokens.shape[1],
            ])

        logprobs = F.log_softmax(logits, axis=-1, dtype=paddle.float32)
        current_logprobs = logprobs[paddle.arange(logprobs.shape[0]),
                                    next_tokens]
        sum_logprobs += current_logprobs * paddle.to_tensor(
            (tokens[:, -1] != self.eot), dtype=paddle.float32)

        next_tokens[tokens[:, -1] == self.eot] = self.eot
        tokens = paddle.concat([tokens, next_tokens[:, None]], axis=-1)

        completed = paddle.all((tokens[:, -1] == self.eot))
        return tokens, completed

    def finalize(self, tokens: paddle.Tensor, sum_logprobs: paddle.Tensor):
        # make sure each sequence has at least one EOT token at the end
        tokens = F.pad(tokens, (0, 1), value=self.eot, data_format="NCL")
        return tokens, sum_logprobs.tolist()


class BeamSearchDecoder(TokenDecoder):
    def __init__(self,
                 beam_size: int,
                 eot: int,
                 inference: Inference,
                 patience: Optional[float]=None):
        self.beam_size = beam_size
        self.eot = eot
        self.inference = inference
        self.patience = patience or 1.0
        self.max_candidates: int = round(beam_size * self.patience)
        self.finished_sequences = None

        assert self.max_candidates > 0, f"Invalid beam size ({beam_size}) or patience ({patience})"

    def reset(self):
        self.finished_sequences = None

    def update(self,
               tokens: paddle.Tensor,
               logits: paddle.Tensor,
               sum_logprobs: paddle.Tensor) -> Tuple[paddle.Tensor, bool]:
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        batch_size = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None:  # for the first update
            self.finished_sequences = [{} for _ in range(batch_size)]

        logprobs = F.log_softmax(logits, axis=-1, dtype=paddle.float32)
        next_tokens, source_indices, finished_sequences = [], [], []
        for i in range(batch_size):
            scores, sources, finished = {}, {}, {}

            # STEP 1: calculate the cumulative log probabilities for possible candidates
            for j in range(self.beam_size):
                idx = i * self.beam_size + j
                prefix = tokens[idx].tolist()
                logprob, token = paddle.topk(
                    logprobs[idx], k=self.beam_size + 1)
                for logprob, token in zip(logprob, token):
                    new_logprob = (sum_logprobs[idx] + logprob).tolist()[0]
                    sequence = tuple(prefix + [token.tolist()[0]])
                    scores[sequence] = new_logprob
                    sources[sequence] = idx

            # STEP 2: rank the candidates and keep the top beam_size sequences for each audio
            saved = 0
            for sequence in sorted(scores, key=scores.get, reverse=True):
                if sequence[-1] == self.eot:
                    finished[sequence] = scores[sequence]
                else:
                    sum_logprobs[len(next_tokens)] = scores[sequence]
                    next_tokens.append(sequence)
                    source_indices.append(sources[sequence])

                    saved += 1
                    if saved == self.beam_size:
                        break

            finished_sequences.append(finished)

        tokens = paddle.to_tensor(next_tokens)
        self.inference.rearrange_kv_cache(source_indices)

        # add newly finished sequences to self.finished_sequences
        assert len(self.finished_sequences) == len(finished_sequences)
        for previously_finished, newly_finished in zip(self.finished_sequences,
                                                       finished_sequences):
            for seq in sorted(
                    newly_finished, key=newly_finished.get, reverse=True):
                if len(previously_finished) >= self.max_candidates:
                    break  # the candidate list is full
                previously_finished[seq] = newly_finished[seq]

        # mark as completed if all audio has enough number of samples
        completed = all(
            len(sequences) >= self.max_candidates
            for sequences in self.finished_sequences)
        return tokens, completed

    def finalize(self,
                 preceding_tokens: paddle.Tensor,
                 sum_logprobs: paddle.Tensor):
        # collect all finished sequences, including patience, and add unfinished ones if not enough
        sum_logprobs = sum_logprobs.cpu()
        for i, sequences in enumerate(self.finished_sequences):
            if len(sequences
                   ) < self.beam_size:  # when not enough sequences are finished
                for j in list(np.argsort(sum_logprobs[i]))[::-1]:
                    sequence = preceding_tokens[i, j].tolist() + [self.eot]
                    sequences[tuple(sequence)] = sum_logprobs[i][j].item()
                    if len(sequences) >= self.beam_size:
                        break

        tokens: List[List[paddle.Tensor]] = [
            [paddle.to_tensor(seq) for seq in sequences.keys()]
            for sequences in self.finished_sequences
        ]
        sum_logprobs: List[List[float]] = [
            list(sequences.values()) for sequences in self.finished_sequences
        ]
        return tokens, sum_logprobs


class LogitFilter:
    def apply(self, logits: paddle.Tensor, tokens: paddle.Tensor) -> None:
        """Apply any filtering or masking to logits in-place

        Parameters
        ----------
        logits : Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        tokens : Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        """
        raise NotImplementedError


class SuppressBlank(LogitFilter):
    def __init__(self, tokenizer: Tokenizer, sample_begin: int):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin

    def apply(self, logits: paddle.Tensor, tokens: paddle.Tensor):
        if tokens.shape[1] == self.sample_begin:
            logits[:, self.tokenizer.encode(" ").input_ids +
                   [self.tokenizer.eot]] = -np.inf


class SuppressTokens(LogitFilter):
    def __init__(self, suppress_tokens: Sequence[int]):
        self.suppress_tokens = list(suppress_tokens)

    def apply(self, logits: paddle.Tensor, tokens: paddle.Tensor):
        logits[:, self.suppress_tokens] = -np.inf


class ApplyTimestampRules(LogitFilter):
    def __init__(self,
                 tokenizer: Tokenizer,
                 sample_begin: int,
                 max_initial_timestamp_index: Optional[int]):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index

    def apply(self, logits: paddle.Tensor, tokens: paddle.Tensor):
        # suppress <|notimestamps|> which is handled by without_timestamps
        if self.tokenizer.no_timestamps is not None:
            logits[:, self.tokenizer.no_timestamps] = -np.inf

        # timestamps have to appear in pairs, except directly before EOT; mask logits accordingly
        for k in range(tokens.shape[0]):
            seq = [t for t in tokens[k, self.sample_begin:].tolist()]
            last_was_timestamp = len(seq) >= 1 and seq[
                -1] >= self.tokenizer.timestamp_begin
            penultimate_was_timestamp = len(seq) < 2 or seq[
                -2] >= self.tokenizer.timestamp_begin

            if last_was_timestamp:
                if penultimate_was_timestamp:  # has to be non-timestamp
                    logits[k, self.tokenizer.timestamp_begin:] = -np.inf
                else:  # cannot be normal text tokens
                    logits[k, :self.tokenizer.eot] = -np.inf

        # apply the `max_initial_timestamp` option
        if tokens.shape[
                1] == self.sample_begin and self.max_initial_timestamp_index is not None:
            last_allowed = self.tokenizer.timestamp_begin + self.max_initial_timestamp_index
            logits[:, last_allowed + 1:] = -np.inf

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = F.log_softmax(logits, axis=-1, dtype=paddle.float32)
        for k in range(tokens.shape[0]):
            # When using paddle.logsumexp on a 32GB Tesla-V100 GPU, we encountered CUDA error 700. 
            # To bypass this issue in CI, we have decomposed the operation into separate steps. 
            # It will raise 2e-6 difference in precision.
            # TODO: revert this after logsumexp been fixed.
            timestamp_logprob = paddle.exp(
                logprobs[k, self.tokenizer.timestamp_begin:])
            timestamp_logprob = paddle.sum(timestamp_logprob, axis=-1)
            timestamp_logprob = paddle.log(timestamp_logprob)
            max_text_token_logprob = paddle.max(
                logprobs[k, :self.tokenizer.timestamp_begin])
            if timestamp_logprob > max_text_token_logprob:
                logits[k, :self.tokenizer.timestamp_begin] = -np.inf


class DecodingTask:
    inference: Inference
    sequence_ranker: SequenceRanker
    decoder: TokenDecoder
    logit_filters: List[LogitFilter]

    def __init__(self,
                 model: "Whisper",
                 options: DecodingOptions,
                 resource_path: str):
        self.model = model

        language = options.language or "en"
        tokenizer = get_tokenizer(
            model.is_multilingual,
            resource_path=resource_path,
            language=language,
            task=options.task)
        self.tokenizer: Tokenizer = tokenizer
        self.options: DecodingOptions = self._verify_options(options)
        self.resource_path: str = resource_path

        self.beam_size: int = options.beam_size or options.best_of or 1
        self.n_ctx: int = model.dims.n_text_ctx
        self.sample_len: int = options.sample_len or model.dims.n_text_ctx // 2

        self.sot_sequence: Tuple[int] = tokenizer.sot_sequence
        if self.options.without_timestamps:
            self.sot_sequence = tokenizer.sot_sequence_including_notimestamps

        self.initial_tokens: Tuple[int] = self._get_initial_tokens()
        self.sample_begin: int = len(self.initial_tokens)
        self.sot_index: int = self.initial_tokens.index(tokenizer.sot)

        # inference: implements the forward pass through the decoder, including kv caching
        self.inference = WhisperInference(model, len(self.initial_tokens))

        # sequence ranker: implements how to rank a group of sampled sequences
        self.sequence_ranker = MaximumLikelihoodRanker(options.length_penalty)

        # decoder: implements how to select the next tokens, given the autoregressive distribution
        if options.beam_size is not None:
            self.decoder = BeamSearchDecoder(options.beam_size, tokenizer.eot,
                                             self.inference, options.patience)
        else:
            self.decoder = GreedyDecoder(options.temperature, tokenizer.eot)

        # logit filters: applies various rules to suppress or penalize certain tokens
        self.logit_filters = []
        if self.options.suppress_blank:
            self.logit_filters.append(
                SuppressBlank(self.tokenizer, self.sample_begin))
        if self.options.suppress_tokens:
            self.logit_filters.append(
                SuppressTokens(self._get_suppress_tokens()))
        if not options.without_timestamps:
            precision = CHUNK_LENGTH / model.dims.n_audio_ctx  # usually 0.02 seconds
            max_initial_timestamp_index = None
            if options.max_initial_timestamp:
                max_initial_timestamp_index = round(
                    self.options.max_initial_timestamp / precision)
            self.logit_filters.append(
                ApplyTimestampRules(tokenizer, self.sample_begin,
                                    max_initial_timestamp_index))

    def _verify_options(self, options: DecodingOptions) -> DecodingOptions:
        if options.beam_size is not None and options.best_of is not None:
            raise ValueError("beam_size and best_of can't be given together")
        if options.temperature == 0:
            if options.best_of is not None:
                raise ValueError(
                    "best_of with greedy sampling (T=0) is not compatible")
        if options.patience is not None and options.beam_size is None:
            raise ValueError("patience requires beam_size to be given")
        if options.length_penalty is not None and not (
                0 <= options.length_penalty <= 1):
            raise ValueError(
                "length_penalty (alpha) should be a value between 0 and 1")

        return options

    def _get_initial_tokens(self) -> Tuple[int]:
        tokens = list(self.sot_sequence)
        prefix = self.options.prefix
        prompt = self.options.prompt

        if prefix:
            prefix_tokens = (
                self.tokenizer.encode(" " + prefix.strip().input_ids)
                if isinstance(prefix, str) else prefix)
            if self.sample_len is not None:
                max_prefix_len = self.n_ctx // 2 - self.sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            tokens = tokens + prefix_tokens

        if prompt:
            prompt_tokens = (
                self.tokenizer.encode(" " + prompt.strip().input_ids)
                if isinstance(prompt, str) else prompt)
            tokens = [self.tokenizer.sot_prev] + prompt_tokens[-(self.n_ctx // 2
                                                                 - 1):] + tokens

        return tuple(tokens)

    def _get_suppress_tokens(self) -> Tuple[int]:
        suppress_tokens = self.options.suppress_tokens

        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(self.tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []  # interpret empty string as an empty list
        else:
            assert isinstance(suppress_tokens,
                              list), "suppress_tokens must be a list"

        suppress_tokens.extend([
            self.tokenizer.sot, self.tokenizer.sot_prev, self.tokenizer.sot_lm
        ])
        if self.tokenizer.no_speech is not None:
            # no-speech probability is collected separately
            suppress_tokens.append(self.tokenizer.no_speech)

        return tuple(sorted(set(suppress_tokens)))

    def _get_audio_features(self, mel: paddle.Tensor):
        #if self.options.fp16:
        #    mel = mel.half()

        if mel.shape[-2:] == (self.model.dims.n_audio_ctx,
                              self.model.dims.n_audio_state):
            # encoded audio features are given; skip audio encoding
            audio_features = mel
        else:
            audio_features = self.model.encoder(mel)

        #if audio_features.dtype != (np.float16 if self.options.fp16 else np.float32):
        #    return TypeError(f"audio_features has an incorrect dtype: {audio_features.dtype}")

        return audio_features

    def _detect_language(self,
                         audio_features: paddle.Tensor,
                         tokens: paddle.Tensor,
                         resource_path: str):
        languages = [self.options.language] * audio_features.shape[0]
        lang_probs = None

        if self.options.language is None or self.options.task == "lang_id":
            lang_tokens, lang_probs = self.model.detect_language(
                audio_features, self.tokenizer, self.resource_path)
            languages = [max(probs, key=probs.get) for probs in lang_probs]
            if self.options.language is None:
                tokens[:, self.sot_index +
                       1] = lang_tokens  # write language tokens

        return languages, lang_probs

    def _main_loop(self, audio_features: paddle.Tensor, tokens: paddle.Tensor):
        assert audio_features.shape[0] == tokens.shape[0]
        n_batch = tokens.shape[0]
        sum_logprobs: paddle.Tensor = paddle.zeros(
            paddle.to_tensor(n_batch), dtype=paddle.float32)
        no_speech_probs = [np.nan] * n_batch

        try:
            for i in range(self.sample_len):
                logits = self.inference.logits(tokens, audio_features)

                if i == 0 and self.tokenizer.no_speech is not None:  # save no_speech_probs
                    probs_at_sot = F.softmax(
                        logits[:, self.sot_index],
                        axis=-1,
                        dtype=paddle.float32)
                    no_speech_probs = probs_at_sot[:, self.tokenizer.
                                                   no_speech].tolist()

                # now we need to consider the logits at the last token only
                logits = logits[:, -1]

                # apply the logit filters, e.g. for suppressing or applying penalty to
                for logit_filter in self.logit_filters:
                    logit_filter.apply(logits, tokens)

                # expand the tokens tensor with the selected next tokens
                tokens, completed = self.decoder.update(tokens, logits,
                                                        sum_logprobs)
                if completed or tokens.shape[-1] > self.n_ctx:
                    break
        finally:
            self.inference.cleanup_caching()

        return tokens, sum_logprobs, no_speech_probs

    @paddle.no_grad()
    def run(self, mel: paddle.Tensor) -> List[DecodingResult]:
        self.decoder.reset()
        tokenizer: Tokenizer = self.tokenizer
        batch_size: int = mel.shape[0]

        audio_features: paddle.Tensor = self._get_audio_features(
            mel)  # encoder forward pass

        tokens: paddle.Tensor
        if batch_size > 1:
            for i in range(batch_size):
                tokens = paddle.concat(
                    x=[
                        paddle.to_tensor([self.initial_tokens]),
                        paddle.to_tensor([self.initial_tokens])
                    ],
                    axis=0)
        elif batch_size == 1:
            tokens = paddle.to_tensor([self.initial_tokens])

        # detect language if requested, overwriting the language token
        languages, language_probs = self._detect_language(
            paddle.to_tensor(audio_features),
            paddle.to_tensor(tokens), self.resource_path)

        if self.options.task == "lang_id":
            return [
                DecodingResult(
                    audio_features=features,
                    language=language,
                    language_probs=probs)
                for features, language, probs in zip(audio_features, languages,
                                                     language_probs)
            ]

        # repeat the audio & text tensors by the group size, for beam search or best-of-n sampling

        audio_features = paddle.repeat_interleave(
            audio_features, self.beam_size, axis=0)
        tokens = paddle.repeat_interleave(tokens, self.beam_size, axis=0)

        # call the main sampling loop
        tokens, sum_logprobs, no_speech_probs = self._main_loop(audio_features,
                                                                tokens)

        # reshape the tensors to have (batch_size, beam_size) as the first two dimensions
        audio_features = audio_features[::self.beam_size]
        no_speech_probs = no_speech_probs[::self.beam_size]
        assert audio_features.shape[0] == len(no_speech_probs) == batch_size

        tokens = tokens.reshape([batch_size, self.beam_size, -1])
        sum_logprobs = sum_logprobs.reshape([batch_size, self.beam_size])

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        tokens: List[List[paddle.Tensor]] = [[
            t[self.sample_begin:(t == tokenizer.eot).nonzero()[0, 0]] for t in s
        ] for s in tokens]

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[
            int]] = [t[i].tolist() for i, t in zip(selected, tokens)]
        texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs: List[
            float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[
            float] = [lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)]

        fields = (texts, languages, tokens, audio_features, avg_logprobs,
                  no_speech_probs)
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(
                f"inconsistent result lengths: {list(map(len, fields))}")

        return [
            DecodingResult(
                audio_features=features,
                language=language,
                tokens=tokens,
                text=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=self.options.temperature,
                compression_ratio=utils.compression_ratio(text), )
            for text, language, tokens, features, avg_logprob, no_speech_prob in
            zip(*fields)
        ]


@paddle.no_grad()
def decode(
        model: "Whisper",
        mel: paddle.Tensor,
        options: DecodingOptions=DecodingOptions(),
        resource_path=str, ) -> Union[DecodingResult, List[DecodingResult]]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model: Whisper
        the Whisper model instance

    mel: paddle.Tensor, shape = (80, 3000) or (*, 80, 3000)
        A tensor containing the Mel spectrogram(s)

    options: DecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments

    Returns
    -------
    result: Union[DecodingResult, List[DecodingResult]]
        The result(s) of decoding contained in `DecodingResult` dataclass instance(s)
    """
    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)

    result = DecodingTask(model, options, resource_path).run(mel)

    if single:
        result = result[0]

    return result


class Whisper(nn.Layer):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer, )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer, )

    def embed_audio(self, mel: paddle.Tensor):
        return self.encoder.forward(mel)

    def logits(self, tokens: paddle.Tensor, audio_features: paddle.Tensor):
        return self.decoder.forward(tokens, audio_features)

    def forward(self, mel: paddle.Tensor,
                tokens: paddle.Tensor) -> Dict[str, paddle.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return paddle.device.get_device()

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict]=None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Layer, paddle.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[
                    1] > self.decoder.positional_embedding.shape[0]:
                cache[
                    module] = output  # save as-is, for the first token or cross attention
            else:
                cache[module] = paddle.concat(
                    [cache[module], output], axis=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Layer):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(
                    layer.key.register_forward_post_hook(save_to_cache))
                hooks.append(
                    layer.value.register_forward_post_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language
    transcribe = transcribe
    decode = decode


def pad_or_trim(array, length: int=N_SAMPLES, *, axis: int=-1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if paddle.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(axis=axis, index=paddle.arange(length))

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = paddle.transpose(array, (1, 0))
            array = F.pad(
                array, [pad for sizes in pad_widths[::-1] for pad in sizes],
                data_format='NLC')
            array = paddle.transpose(array, (1, 0))
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = paddle.transpose(array, (1, 0))
            array = np.pad(array, pad_widths)
            array = paddle.transpose(array, (1, 0))

    return array


def hann_window(n_fft: int=N_FFT):
    """
    hanning window
    n_fft:  The number of frequency components of the discrete Fourier transform.
    """
    return paddle.to_tensor(
        [0.5 - 0.5 * np.cos(2 * np.pi * n / n_fft) for n in range(n_fft)],
        dtype=paddle.float32)


@lru_cache(maxsize=None)
def mel_filters(resource_path: str, n_mels: int=N_MELS) -> paddle.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(os.path.join(resource_path, "assets", "mel_filters.npz")) as f:
        return paddle.to_tensor(f[f"mel_{n_mels}"])


def log_mel_spectrogram(audio: Union[str, np.ndarray, paddle.Tensor],
                        n_mels: int=N_MELS,
                        resource_path: str=None):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, paddle.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    Returns
    -------
    paddle.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not paddle.is_tensor(audio):
        if isinstance(audio, str):
            audio, _ = soundfile.read(audio, dtype="float32", always_2d=True)
            audio = audio[:, 0]
            logger.info(f"audio shape: {audio.shape}")
        audio = paddle.to_tensor(audio)

    window = hann_window(N_FFT)
    stft = paddle.signal.stft(audio, N_FFT, HOP_LENGTH, window=window)

    magnitudes = stft[:, :-1].abs()**2

    filters = mel_filters(resource_path, n_mels)
    mel_spec = filters @ magnitudes
    mel_spec = paddle.to_tensor(mel_spec.numpy().tolist())

    log_spec = paddle.clip(mel_spec, min=1e-10).log10()
    log_spec = paddle.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec
