# MIT License, Copyright (c) 2022 OpenAI.
# Copyright (c) 2022 PaddlePaddle Authors and . All Rights Reserved.
# 
# Modified from OpenAI Whisper 2022 (https://github.com/openai/whisper/whisper/__init__.py)

import hashlib
import io
import os
import urllib
import warnings
from typing import List, Optional, Union
from more_itertools import padded

import paddle
from tqdm import tqdm

from paddlespeech.s2t.models.whisper.audio import log_mel_spectrogram, pad_or_trim
from paddlespeech.s2t.models.whisper.decoding import DecodingOptions, DecodingResult, decode, detect_language
from paddlespeech.s2t.models.whisper.model import Whisper, ModelDimensions
from paddlespeech.s2t.models.whisper.transcribe import transcribe

_MODELS = {
    "large" : "https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221108/large.model.pdparams"
}
_MODELS_sha256 = {
    "large" : "589a2229582cc9173091f2481bba2cc8228997502ac75cbb0be6d874e8433d0f"
}

def _download(model_key: str, root: str, in_memory: bool) -> Union[bytes, str]:
    os.makedirs(root, exist_ok=True)

    expected_sha256 = _MODELS_sha256[model_key]
    url = _MODELS[model_key]
    download_target = os.path.join(root, os.path.basename(url))

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        model_bytes = open(download_target, "rb").read()
        if hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
            return model_bytes if in_memory else download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    model_bytes = open(download_target, "rb").read()
    if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model.")

    return model_bytes if in_memory else download_target


def available_models() -> List[str]:
    """Returns the names of available models"""
    return list(_MODELS.keys())


