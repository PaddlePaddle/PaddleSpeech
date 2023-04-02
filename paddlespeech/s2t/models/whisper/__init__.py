# MIT License, Copyright (c) 2022 OpenAI.
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Modified from OpenAI Whisper 2022 (https://github.com/openai/whisper/whisper/__init__.py)
from paddlespeech.s2t.models.whisper.whipser import decode
from paddlespeech.s2t.models.whisper.whipser import DecodingOptions
from paddlespeech.s2t.models.whisper.whipser import DecodingResult
from paddlespeech.s2t.models.whisper.whipser import detect_language
from paddlespeech.s2t.models.whisper.whipser import log_mel_spectrogram
from paddlespeech.s2t.models.whisper.whipser import ModelDimensions
from paddlespeech.s2t.models.whisper.whipser import transcribe
from paddlespeech.s2t.models.whisper.whipser import Whisper
