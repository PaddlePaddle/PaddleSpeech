# MIT License, Copyright (c) 2022 OpenAI.
# Copyright (c) 2022 PaddlePaddle Authors and . All Rights Reserved.
# 
# Modified from OpenAI Whisper 2022 (https://github.com/openai/whisper/whisper/transcribe.py)
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

import numpy as np
import paddle
import tqdm

from paddlespeech.s2t.models.whisper.audio import HOP_LENGTH
from paddlespeech.s2t.models.whisper.audio import log_mel_spectrogram
from paddlespeech.s2t.models.whisper.audio import N_FRAMES
from paddlespeech.s2t.models.whisper.audio import pad_or_trim
from paddlespeech.s2t.models.whisper.audio import SAMPLE_RATE
from paddlespeech.s2t.models.whisper.decoding import DecodingOptions
from paddlespeech.s2t.models.whisper.decoding import DecodingResult
from paddlespeech.s2t.models.whisper.tokenizer import get_tokenizer
from paddlespeech.s2t.models.whisper.tokenizer import LANGUAGES
from paddlespeech.s2t.models.whisper.utils import exact_div
from paddlespeech.s2t.models.whisper.utils import format_timestamp

if TYPE_CHECKING:
    from paddlespeech.s2t.models.whisper.model import Whisper


def transcribe(
        model: "Whisper",
        audio: Union[str, np.ndarray, paddle.Tensor],
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

    audio: Union[str, np.ndarray, torch.Tensor]
        The path to the audio file to open, or the audio waveform

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

    mel = log_mel_spectrogram(audio)

    if decode_options.get("language", None) is None:
        if not model.is_multilingual:
            decode_options["language"] = "en"
        else:
            if verbose:
                print(
                    "Detecting language using up to the first 30 seconds. Use `--language` to specify the language"
                )
            segment = pad_or_trim(mel, N_FRAMES)
            _, probs = model.detect_language(segment)
            decode_options["language"] = max(probs, key=probs.get)
            if verbose is not None:
                print(
                    f"Detected language: {LANGUAGES[decode_options['language']].title()}"
                )

    language = decode_options["language"]
    task = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(
        model.is_multilingual, language=language, task=task)

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
            decode_result = model.decode(segment, options)

            needs_fallback = False
            if compression_ratio_threshold is not None and decode_result.compression_ratio > compression_ratio_threshold:
                needs_fallback = True  # too repetitive
            if logprob_threshold is not None and decode_result.avg_logprob < logprob_threshold:
                needs_fallback = True  # average log probability is too low

            if not needs_fallback:
                break

        return decode_result

    seek = 0
    input_stride = exact_div(
        N_FRAMES, model.dims.n_audio_ctx)  # mel frames per output token: 2
    time_precision = (input_stride * HOP_LENGTH /
                      SAMPLE_RATE)  # time per output token: 0.02 (seconds)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    initial_prompt = decode_options.pop("initial_prompt", None) or []
    if initial_prompt:
        initial_prompt = tokenizer.encode(" " + initial_prompt.strip())
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
                f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}"
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
