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
import io

import numpy as np

from paddlespeech.s2t.frontend.augmentor.augmentation import AugmentationPipeline
from paddlespeech.s2t.frontend.featurizer.speech_featurizer import SpeechFeaturizer
from paddlespeech.s2t.frontend.normalizer import FeatureNormalizer
from paddlespeech.s2t.frontend.speech import SpeechSegment
from paddlespeech.s2t.frontend.utility import IGNORE_ID
from paddlespeech.s2t.frontend.utility import TarLocalData
from paddlespeech.s2t.io.reader import LoadInputsAndTargets
from paddlespeech.s2t.io.utility import pad_list
from paddlespeech.s2t.utils.log import Log

__all__ = ["SpeechCollator", "TripletSpeechCollator"]

logger = Log(__name__).getlog()


def _tokenids(text, keep_transcription_text):
    # for training text is token ids
    tokens = text  # token ids

    if keep_transcription_text:
        # text is string, convert to unicode ord
        assert isinstance(text, str), (type(text), text)
        tokens = [ord(t) for t in text]

    tokens = np.array(tokens, dtype=np.int64)
    return tokens


class SpeechCollatorBase():
    def __init__(
            self,
            aug_file,
            mean_std_filepath,
            vocab_filepath,
            spm_model_prefix,
            random_seed=0,
            unit_type="char",
            spectrum_type='linear',  # 'linear', 'mfcc', 'fbank'
            feat_dim=0,  # 'mfcc', 'fbank'
            delta_delta=False,  # 'mfcc', 'fbank'
            stride_ms=10.0,  # ms
            window_ms=20.0,  # ms
            n_fft=None,  # fft points
            max_freq=None,  # None for samplerate/2
            target_sample_rate=16000,  # target sample rate
            use_dB_normalization=True,
            target_dB=-20,
            dither=1.0,
            keep_transcription_text=True):
        """SpeechCollator Collator

        Args:
            unit_type(str): token unit type, e.g. char, word, spm
            vocab_filepath (str): vocab file path.
            mean_std_filepath (str): mean and std file path, which suffix is *.npy
            spm_model_prefix (str): spm model prefix, need if `unit_type` is spm.
            augmentation_config (str, optional): augmentation json str. Defaults to '{}'.
            stride_ms (float, optional): stride size in ms. Defaults to 10.0.
            window_ms (float, optional): window size in ms. Defaults to 20.0.
            n_fft (int, optional): fft points for rfft. Defaults to None.
            max_freq (int, optional): max cut freq. Defaults to None.
            target_sample_rate (int, optional): target sample rate which used for training. Defaults to 16000.
            spectrum_type (str, optional): 'linear', 'mfcc' or 'fbank'. Defaults to 'linear'.
            feat_dim (int, optional): audio feature dim, using by 'mfcc' or 'fbank'. Defaults to None.
            delta_delta (bool, optional): audio feature with delta-delta, using by 'fbank' or 'mfcc'. Defaults to False.
            use_dB_normalization (bool, optional): do dB normalization. Defaults to True.
            target_dB (int, optional): target dB. Defaults to -20.
            random_seed (int, optional): for random generator. Defaults to 0.
            keep_transcription_text (bool, optional): True, when not in training mode, will not do tokenizer; Defaults to False.
            if ``keep_transcription_text`` is False, text is token ids else is raw string.

        Do augmentations
        Padding audio features with zeros to make them have the same shape (or
        a user-defined shape) within one batch.
        """
        self.keep_transcription_text = keep_transcription_text
        self.train_mode = not keep_transcription_text

        self.stride_ms = stride_ms
        self.window_ms = window_ms
        self.feat_dim = feat_dim

        self.loader = LoadInputsAndTargets()

        # only for tar filetype
        self._local_data = TarLocalData(tar2info={}, tar2object={})

        self.augmentation = AugmentationPipeline(
            preprocess_conf=aug_file.read(), random_seed=random_seed)

        self._normalizer = FeatureNormalizer(
            mean_std_filepath) if mean_std_filepath else None

        self._speech_featurizer = SpeechFeaturizer(
            unit_type=unit_type,
            vocab_filepath=vocab_filepath,
            spm_model_prefix=spm_model_prefix,
            spectrum_type=spectrum_type,
            feat_dim=feat_dim,
            delta_delta=delta_delta,
            stride_ms=stride_ms,
            window_ms=window_ms,
            n_fft=n_fft,
            max_freq=max_freq,
            target_sample_rate=target_sample_rate,
            use_dB_normalization=use_dB_normalization,
            target_dB=target_dB,
            dither=dither)

        self.feature_size = self._speech_featurizer.audio_feature.feature_size
        self.text_feature = self._speech_featurizer.text_feature
        self.vocab_dict = self.text_feature.vocab_dict
        self.vocab_list = self.text_feature.vocab_list
        self.vocab_size = self.text_feature.vocab_size

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
        filetype = self.loader.file_type(audio_file)

        if filetype != 'sound':
            spectrum = self.loader._get_from_loader(audio_file, filetype)
            feat_dim = spectrum.shape[1]
            assert feat_dim == self.feat_dim, f"expect feat dim {self.feat_dim}, but got {feat_dim}"

            if self.keep_transcription_text:
                transcript_part = transcript
            else:
                text_ids = self.text_feature.featurize(transcript)
                transcript_part = text_ids
        else:
            # read audio
            speech_segment = SpeechSegment.from_file(audio_file,
                                                     transcript,
                                                     infos=self._local_data)
            # audio augment
            self.augmentation.transform_audio(speech_segment)

            # extract speech feature
            spectrum, transcript_part = self._speech_featurizer.featurize(
                speech_segment, self.keep_transcription_text)
            # CMVN spectrum
            if self._normalizer:
                spectrum = self._normalizer.apply(spectrum)

        # spectrum augment
        spectrum = self.augmentation.transform_feature(spectrum)
        return spectrum, transcript_part

    def __call__(self, batch):
        """batch examples

        Args:
            batch (List[Dict]): batch is [dict(audio, text, ...)]
                audio (np.ndarray) shape (T, D)
                text (List[int] or str): shape (U,)

        Returns:
            tuple(utts, xs_pad, ilens, ys_pad, olens): batched data.
                utts: (B,)
                xs_pad : (B, Tmax, D)
                ilens: (B,)
                ys_pad : (B, Umax)
                olens: (B,)
        """
        audios = []
        audio_lens = []
        texts = []
        text_lens = []
        utts = []
        tids = []  # tokenids

        for idx, item in enumerate(batch):
            utts.append(item['utt'])

            audio = item['input'][0]['feat']
            text = item['output'][0]['text']
            audio, text = self.process_utterance(audio, text)

            audios.append(audio)  # [T, D]
            audio_lens.append(audio.shape[0])

            tokens = _tokenids(text, self.keep_transcription_text)
            texts.append(tokens)
            text_lens.append(tokens.shape[0])

        #[B, T, D]
        xs_pad = pad_list(audios, 0.0).astype(np.float32)
        ilens = np.array(audio_lens).astype(np.int64)
        ys_pad = pad_list(texts, IGNORE_ID).astype(np.int64)
        olens = np.array(text_lens).astype(np.int64)
        return utts, xs_pad, ilens, ys_pad, olens


class SpeechCollator(SpeechCollatorBase):
    @classmethod
    def from_config(cls, config):
        """Build a SpeechCollator object from a config.

        Args:
            config (yacs.config.CfgNode): configs object.

        Returns:
            SpeechCollator: collator object.
        """
        assert 'augmentation_config' in config
        assert 'keep_transcription_text' in config
        assert 'mean_std_filepath' in config
        assert 'vocab_filepath' in config
        assert 'spectrum_type' in config
        assert 'n_fft' in config
        assert config

        if isinstance(config.augmentation_config, (str, bytes)):
            if config.augmentation_config:
                aug_file = io.open(config.augmentation_config,
                                   mode='r',
                                   encoding='utf8')
            else:
                aug_file = io.StringIO(initial_value='{}', newline='')
        else:
            aug_file = config.augmentation_config
            assert isinstance(aug_file, io.StringIO)

        speech_collator = cls(
            aug_file=aug_file,
            random_seed=0,
            mean_std_filepath=config.mean_std_filepath,
            unit_type=config.unit_type,
            vocab_filepath=config.vocab_filepath,
            spm_model_prefix=config.spm_model_prefix,
            spectrum_type=config.spectrum_type,
            feat_dim=config.feat_dim,
            delta_delta=config.delta_delta,
            stride_ms=config.stride_ms,
            window_ms=config.window_ms,
            n_fft=config.n_fft,
            max_freq=config.max_freq,
            target_sample_rate=config.target_sample_rate,
            use_dB_normalization=config.use_dB_normalization,
            target_dB=config.target_dB,
            dither=config.dither,
            keep_transcription_text=config.keep_transcription_text)
        return speech_collator


class TripletSpeechCollator(SpeechCollator):
    def process_utterance(self, audio_file, translation, transcript):
        """Load, augment, featurize and normalize for speech data.

        :param audio_file: Filepath or file object of audio file.
        :type audio_file: str | file
        :param translation: translation text.
        :type translation: str
        :return: Tuple of audio feature tensor and data of translation part,
                    where translation part could be token ids or text.
        :rtype: tuple of (2darray, list)
        """
        spectrum, translation_part = super().process_utterance(
            audio_file, translation)
        transcript_part = self._speech_featurizer.text_featurize(
            transcript, self.keep_transcription_text)
        return spectrum, translation_part, transcript_part

    def __call__(self, batch):
        """batch examples

        Args:
            batch (List[Dict]): batch is [dict(audio, text, ...)]
                audio (np.ndarray) shape (T, D)
                text (List[int] or str): shape (U,)

        Returns:
            tuple(utts, xs_pad, ilens, ys_pad, olens): batched data.
                utts: (B,)
                xs_pad : (B, Tmax, D)
                ilens: (B,)
                ys_pad : [(B, Umax), (B, Umax)]
                olens: [(B,), (B,)]
        """
        utts = []
        audios = []
        audio_lens = []
        translation_text = []
        translation_text_lens = []
        transcription_text = []
        transcription_text_lens = []

        for idx, item in enumerate(batch):
            utts.append(item['utt'])

            audio = item['input'][0]['feat']
            translation = item['output'][0]['text']
            transcription = item['output'][1]['text']

            audio, translation, transcription = self.process_utterance(
                audio, translation, transcription)

            audios.append(audio)  # [T, D]
            audio_lens.append(audio.shape[0])

            tokens = [[], []]
            for idx, text in enumerate([translation, transcription]):
                tokens[idx] = _tokenids(text, self.keep_transcription_text)

            translation_text.append(tokens[0])
            translation_text_lens.append(tokens[0].shape[0])
            transcription_text.append(tokens[1])
            transcription_text_lens.append(tokens[1].shape[0])

        xs_pad = pad_list(audios, 0.0).astype(np.float32)  #[B, T, D]
        ilens = np.array(audio_lens).astype(np.int64)

        padded_translation = pad_list(translation_text,
                                      IGNORE_ID).astype(np.int64)
        translation_lens = np.array(translation_text_lens).astype(np.int64)

        padded_transcription = pad_list(transcription_text,
                                        IGNORE_ID).astype(np.int64)
        transcription_lens = np.array(transcription_text_lens).astype(np.int64)

        ys_pad = (padded_translation, padded_transcription)
        olens = (translation_lens, transcription_lens)
        return utts, xs_pad, ilens, ys_pad, olens
