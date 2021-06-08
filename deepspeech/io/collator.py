# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from deepspeech.frontend.utility import IGNORE_ID
from deepspeech.io.utility import pad_sequence
from deepspeech.utils.log import Log
from deepspeech.frontend.augmentor.augmentation import AugmentationPipeline
from deepspeech.frontend.featurizer.speech_featurizer import SpeechFeaturizer
from deepspeech.frontend.normalizer import FeatureNormalizer
from deepspeech.frontend.speech import SpeechSegment
import io
import time

__all__ = ["SpeechCollator"]

logger = Log(__name__).getlog()

# namedtupe need global for pickle.
TarLocalData = namedtuple('TarLocalData', ['tar2info', 'tar2object'])

class SpeechCollator():
    def __init__(self, config, keep_transcription_text=True):
        """
        Padding audio features with zeros to make them have the same shape (or
        a user-defined shape) within one bach.

        if ``keep_transcription_text`` is False, text is token ids else is raw string.
        """
        self._keep_transcription_text = keep_transcription_text

        if isinstance(config.data.augmentation_config, (str, bytes)):
            if config.data.augmentation_config:
                aug_file = io.open(
                    config.data.augmentation_config, mode='r', encoding='utf8')
            else:
                aug_file = io.StringIO(initial_value='{}', newline='')
        else:
            aug_file = config.data.augmentation_config
            assert isinstance(aug_file, io.StringIO)

        self._local_data = TarLocalData(tar2info={}, tar2object={}）
        self._augmentation_pipeline = AugmentationPipeline(
            augmentation_config=aug_file.read(), 
            random_seed=config.data.random_seed)
        
        self._normalizer = FeatureNormalizer(
            config.data.mean_std_filepath) if config.data.mean_std_filepath else None

        self._stride_ms = config.data.stride_ms
        self._target_sample_rate = config.data.target_sample_rate

        self._speech_featurizer = SpeechFeaturizer(
            unit_type=config.data.unit_type,
            vocab_filepath=config.data.vocab_filepath,
            spm_model_prefix=config.data.spm_model_prefix,
            specgram_type=config.data.specgram_type,
            feat_dim=config.data.feat_dim,
            delta_delta=config.data.delta_delta,
            stride_ms=config.data.stride_ms,
            window_ms=config.data.window_ms,
            n_fft=config.data.n_fft,
            max_freq=config.data.max_freq,
            target_sample_rate=config.data.target_sample_rate,
            use_dB_normalization=config.data.use_dB_normalization,
            target_dB=config.data.target_dB,
            dither=config.data.dither)

    def _parse_tar(self, file):
        """Parse a tar file to get a tarfile object
        and a map containing tarinfoes
        """
        result = {}
        f = tarfile.open(file)
        for tarinfo in f.getmembers():
            result[tarinfo.name] = tarinfo
        return f, result

    def _subfile_from_tar(self, file):
        """Get subfile object from tar.

        It will return a subfile object from tar file
        and cached tar file info for next reading request.
        """
        tarpath, filename = file.split(':', 1)[1].split('#', 1)
        if 'tar2info' not in self._local_data.__dict__:
            self._local_data.tar2info = {}
        if 'tar2object' not in self._local_data.__dict__:
            self._local_data.tar2object = {}
        if tarpath not in self._local_data.tar2info:
            object, infoes = self._parse_tar(tarpath)
            self._local_data.tar2info[tarpath] = infoes
            self._local_data.tar2object[tarpath] = object
        return self._local_data.tar2object[tarpath].extractfile(
            self._local_data.tar2info[tarpath][filename])

    def process_utterance(self, audio_file, transcript):
        """Load, augment, featurize and normalize for speech data.

        :param audio_file: Filepath or file object of audio file.
        :type audio_file: str | file
        :param transcript: Transcription text.
        :type transcript: str
        :return: Tuple of audio feature tensor and data of transcription part,
                 where transcription part could be token ids or text.
        :rtype: tuple of (2darray, list)
        """
        start_time = time.time()
        if isinstance(audio_file, str) and audio_file.startswith('tar:'):
            speech_segment = SpeechSegment.from_file(
                self._subfile_from_tar(audio_file), transcript)
        else:
            speech_segment = SpeechSegment.from_file(audio_file, transcript)
        load_wav_time = time.time() - start_time
        #logger.debug(f"load wav time: {load_wav_time}")

        # audio augment
        start_time = time.time()
        self._augmentation_pipeline.transform_audio(speech_segment)
        audio_aug_time = time.time() - start_time
        #logger.debug(f"audio augmentation time: {audio_aug_time}")

        start_time = time.time()
        specgram, transcript_part = self._speech_featurizer.featurize(
            speech_segment, self._keep_transcription_text)
        if self._normalizer:
            specgram = self._normalizer.apply(specgram)
        feature_time = time.time() - start_time
        #logger.debug(f"audio & test feature time: {feature_time}")

        # specgram augment
        start_time = time.time()
        specgram = self._augmentation_pipeline.transform_feature(specgram)
        feature_aug_time = time.time() - start_time
        #logger.debug(f"audio feature augmentation time: {feature_aug_time}")
        return specgram, transcript_part

    def __call__(self, batch):
        """batch examples

        Args:
            batch ([List]): batch is (audio, text)
                audio (np.ndarray) shape (D, T)
                text (List[int] or str): shape (U,)

        Returns:
            tuple(audio, text, audio_lens, text_lens): batched data.
                audio : (B, Tmax, D)
                audio_lens: (B)
                text : (B, Umax)
                text_lens: (B)
        """
        audios = []
        audio_lens = []
        texts = []
        text_lens = []
        utts = []
        for utt, audio, text in batch:
            audio, text = self.process_utterance(audio, text)
            #utt
            utts.append(utt)
            # audio
            audios.append(audio.T)  # [T, D]
            audio_lens.append(audio.shape[1])
            # text
            # for training, text is token ids
            # else text is string, convert to unicode ord
            tokens = []
            if self._keep_transcription_text:
                assert isinstance(text, str), (type(text), text)
                tokens = [ord(t) for t in text]
            else:
                tokens = text  # token ids
            tokens = tokens if isinstance(tokens, np.ndarray) else np.array(
                tokens, dtype=np.int64)
            texts.append(tokens)
            text_lens.append(tokens.shape[0])

        padded_audios = pad_sequence(
            audios, padding_value=0.0).astype(np.float32)  #[B, T, D]
        audio_lens = np.array(audio_lens).astype(np.int64)
        padded_texts = pad_sequence(
            texts, padding_value=IGNORE_ID).astype(np.int64)
        text_lens = np.array(text_lens).astype(np.int64)
        return utts, padded_audios, audio_lens, padded_texts, text_lens
