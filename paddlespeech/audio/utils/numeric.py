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
import numpy as np


def pcm16to32(audio: np.ndarray) -> np.ndarray:
    """pcm int16 to float32

    Args:
        audio (np.ndarray): Waveform with dtype of int16.

    Returns:
        np.ndarray: Waveform with dtype of float32.
    """
    if audio.dtype == np.int16:
        audio = audio.astype("float32")
        bits = np.iinfo(np.int16).bits
        audio = audio / (2**(bits - 1))
    return audio
