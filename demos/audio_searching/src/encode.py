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
import librosa
import numpy as np
from logs import LOGGER


def get_audio_embedding(path):
    """
    Use vpr_inference to generate embedding of audio
    """
    try:
        RESAMPLE_RATE = 16000
        audio, _ = librosa.load(path, sr=RESAMPLE_RATE, mono=True)

        # TODO add infer/python interface to get embedding, now fake it by rand
        # vpr = ECAPATDNN(checkpoint_path=None, device='cuda')
        # embedding = vpr.inference(audio)

        embedding = np.random.rand(1, 2048)
        embedding = embedding / np.linalg.norm(embedding)
        embedding = embedding.tolist()[0]
        return embedding
    except Exception as e:
        LOGGER.error(f"Error with embedding:{e}")
        return None
