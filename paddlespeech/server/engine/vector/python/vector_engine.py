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
import io
from collections import OrderedDict

import numpy as np
import paddle

from paddlespeech.audio.soundfile_backend import soundfile_load as load_audio
from paddlespeech.audio.compliance.librosa import melspectrogram
from paddlespeech.cli.log import logger
from paddlespeech.cli.vector.infer import VectorExecutor
from paddlespeech.server.engine.base_engine import BaseEngine
from paddlespeech.vector.io.batch import feature_normalize


class PaddleVectorConnectionHandler:
    def __init__(self, vector_engine):
        """The PaddleSpeech Vector Server Connection Handler
           This connection process every server request
        Args:
            vector_engine (VectorEngine): The Vector engine
        """
        super().__init__()
        logger.debug(
            "Create PaddleVectorConnectionHandler to process the vector request")
        self.vector_engine = vector_engine
        self.executor = self.vector_engine.executor
        self.task = self.vector_engine.executor.task
        self.model = self.vector_engine.executor.model
        self.config = self.vector_engine.executor.config

        self._inputs = OrderedDict()
        self._outputs = OrderedDict()

    @paddle.no_grad()
    def run(self, audio_data, task="spk"):
        """The connection process the http request audio

        Args:
            audio_data (bytes): base64.b64decode

        Returns:
            str: the punctuation text
        """
        logger.debug(
            f"start to extract the do vector {self.task} from the http request")
        if self.task == "spk" and task == "spk":
            embedding = self.extract_audio_embedding(audio_data)
            return embedding
        else:
            logger.error(
                "The request task is not matched with server model task")
            logger.error(
                f"The server model task is: {self.task}, but the request task is: {task}"
            )

        return np.array([
            0.0,
        ])

    @paddle.no_grad()
    def get_enroll_test_score(self, enroll_audio, test_audio):
        """Get the enroll and test audio score

        Args:
            enroll_audio (str): the base64 format enroll audio
            test_audio (str): the base64 format test audio

        Returns:
            float: the score between enroll and test audio
        """
        logger.debug("start to extract the enroll audio embedding")
        enroll_emb = self.extract_audio_embedding(enroll_audio)

        logger.debug("start to extract the test audio embedding")
        test_emb = self.extract_audio_embedding(test_audio)

        logger.debug(
            "start to get the score between the enroll and test embedding")
        score = self.executor.get_embeddings_score(enroll_emb, test_emb)

        logger.debug(f"get the enroll vs test score: {score}")
        return score

    @paddle.no_grad()
    def extract_audio_embedding(self, audio: str, sample_rate: int=16000):
        """extract the audio embedding

        Args:
            audio (str): the audio data
            sample_rate (int, optional): the audio sample rate. Defaults to 16000.
        """
        # we can not reuse the cache io.BytesIO(audio) data, 
        # because the soundfile will change the io.BytesIO(audio) to the end
        # thus we should convert the base64 string to io.BytesIO when we need the audio data
        if not self.executor._check(io.BytesIO(audio), sample_rate):
            logger.debug("check the audio sample rate occurs error")
            return np.array([0.0])

        waveform, sr = load_audio(io.BytesIO(audio))
        logger.debug(
            f"load the audio sample points, shape is: {waveform.shape}")

        # stage 2: get the audio feat
        # Note: Now we only support fbank feature
        try:
            feats = melspectrogram(
                x=waveform,
                sr=self.config.sr,
                n_mels=self.config.n_mels,
                window_size=self.config.window_size,
                hop_length=self.config.hop_size)
            logger.debug(f"extract the audio feats, shape is: {feats.shape}")
        except Exception as e:
            logger.error(f"feats occurs exception {e}")
            sys.exit(-1)

        feats = paddle.to_tensor(feats).unsqueeze(0)
        # in inference period, the lengths is all one without padding
        lengths = paddle.ones([1])

        # stage 3: we do feature normalize,
        #          Now we assume that the feats must do normalize
        feats = feature_normalize(feats, mean_norm=True, std_norm=False)

        # stage 4: store the feats and length in the _inputs,
        #          which will be used in other function
        logger.info(f"feats shape: {feats.shape}")
        logger.info("audio extract the feats success")

        logger.info("start to extract the audio embedding")
        embedding = self.model.backbone(feats, lengths).squeeze().numpy()
        logger.info(f"embedding size: {embedding.shape}")

        return embedding


class VectorServerExecutor(VectorExecutor):
    def __init__(self):
        """The wrapper for TextEcutor
        """
        super().__init__()
        pass


class VectorEngine(BaseEngine):
    def __init__(self):
        """The Vector Engine
        """
        super(VectorEngine, self).__init__()
        logger.debug("Create the VectorEngine Instance")

    def init(self, config: dict):
        """Init the Vector Engine

        Args:
            config (dict): The server configuation

        Returns:
            bool: The engine instance flag
        """
        logger.debug("Init the vector engine")
        try:
            self.config = config
            if self.config.device:
                self.device = self.config.device
            else:
                self.device = paddle.get_device()

            paddle.set_device(self.device)
            logger.debug(f"Vector Engine set the device: {self.device}")
        except BaseException as e:
            logger.error(
                "Set device failed, please check if device is already used and the parameter 'device' in the yaml file"
            )
            logger.error("Initialize Vector server engine Failed on device: %s."
                         % (self.device))
            return False

        self.executor = VectorServerExecutor()

        self.executor._init_from_path(
            model_type=config.model_type,
            cfg_path=config.cfg_path,
            ckpt_path=config.ckpt_path,
            task=config.task)

        logger.info(
            "Initialize Vector server engine successfully on device: %s." %
            (self.device))
        return True
