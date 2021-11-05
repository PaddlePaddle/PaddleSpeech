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
import struct
from pathlib import Path
from warnings import warn

import librosa
import numpy as np
from scipy.ndimage.morphology import binary_dilation

try:
    import webrtcvad
except ModuleNotFoundError:
    warn("Unable to import 'webrtcvad'."
         "This package enables noise removal and is recommended.")
    webrtcvad = None

INT16_MAX = (2**15) - 1


def normalize_volume(wav, target_dBFS, increase_only=False,
                     decrease_only=False):
    # this function implements Loudness normalization, instead of peak
    # normalization, See https://en.wikipedia.org/wiki/Audio_normalization
    # dBFS: Decibels relative to full scale
    # See https://en.wikipedia.org/wiki/DBFS for more details
    # for 16Bit PCM audio, minimal level is -96dB
    # compute the mean dBFS and adjust to target dBFS, with by increasing
    # or decreasing
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav**2))
    if dBFS_change < 0 and increase_only:
        return wav
    if dBFS_change > 0 and decrease_only:
        return wav
    gain = 10**(dBFS_change / 20)
    return wav * gain


def trim_long_silences(wav,
                       vad_window_length: int,
                       vad_moving_average_width: int,
                       vad_max_silence_length: int,
                       sampling_rate: int):
    """
    Ensures that segments without voice in the waveform remain no longer than a
    threshold determined by the VAD parameters in params.py.
    Parameters
    ----------
    wav : np.array
        the raw waveform as a numpy array of floats
    Returns
    ----------
    np.array
        the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000

    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]

    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav),
                           *(np.round(wav * INT16_MAX)).astype(np.int16))

    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(
            vad.is_speech(
                pcm_wave[window_start * 2:window_end * 2],
                sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)

    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array,
                                       np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width

    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(bool)

    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask,
                                 np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)

    return wav[audio_mask]


def compute_partial_slices(n_samples: int,
                           partial_utterance_n_frames: int,
                           hop_length: int,
                           min_pad_coverage: float=0.75,
                           overlap: float=0.5):
    """
    Computes where to split an utterance waveform and its corresponding mel spectrogram to obtain
    partial utterances of <partial_utterance_n_frames> each. Both the waveform and the mel
    spectrogram slices are returned, so as to make each partial utterance waveform correspond to
    its spectrogram. This function assumes that the mel spectrogram parameters used are those
    defined in params_data.py.

    The returned ranges may be indexing further than the length of the waveform. It is
    recommended that you pad the waveform with zeros up to wave_slices[-1].stop.
    Parameters
    ----------
    n_samples : int
        the number of samples in the waveform.
    partial_utterance_n_frames : int
        the number of mel spectrogram frames in each partial utterance.

    min_pad_coverage : int 
        when reaching the last partial utterance, it may or may not have enough frames.
        If at least <min_pad_coverage> of <partial_utterance_n_frames> are present,
        then the last partial utterance will be considered, as if we padded the audio. Otherwise,
        it will be discarded, as if we trimmed the audio. If there aren't enough frames for 1 partial
        utterance, this parameter is ignored so that the function always returns at least 1 slice.
    overlap : float
        by how much the partial utterance should overlap. If set to 0, the partial utterances are entirely disjoint.
    Returns
    ----------
        the waveform slices and mel spectrogram slices as lists of array slices. 
        Index respectively the waveform and the mel spectrogram with these slices to obtain the partialutterances.
    """
    assert 0 <= overlap < 1
    assert 0 < min_pad_coverage <= 1

    # librosa's function to compute num_frames from num_samples
    n_frames = int(np.ceil((n_samples + 1) / hop_length))
    # frame shift between ajacent partials
    frame_step = max(1,
                     int(np.round(partial_utterance_n_frames * (1 - overlap))))

    # Compute the slices
    wav_slices, mel_slices = [], []
    steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + partial_utterance_n_frames])
        wav_range = mel_range * hop_length
        mel_slices.append(slice(*mel_range))
        wav_slices.append(slice(*wav_range))

    # Evaluate whether extra padding is warranted or not
    last_wav_range = wav_slices[-1]
    coverage = (n_samples - last_wav_range.start) / (
        last_wav_range.stop - last_wav_range.start)
    if coverage < min_pad_coverage and len(mel_slices) > 1:
        mel_slices = mel_slices[:-1]
        wav_slices = wav_slices[:-1]

    return wav_slices, mel_slices


class SpeakerVerificationPreprocessor(object):
    def __init__(self,
                 sampling_rate: int,
                 audio_norm_target_dBFS: float,
                 vad_window_length,
                 vad_moving_average_width,
                 vad_max_silence_length,
                 mel_window_length,
                 mel_window_step,
                 n_mels,
                 partial_n_frames: int,
                 min_pad_coverage: float=0.75,
                 partial_overlap_ratio: float=0.5):
        self.sampling_rate = sampling_rate
        self.audio_norm_target_dBFS = audio_norm_target_dBFS

        self.vad_window_length = vad_window_length
        self.vad_moving_average_width = vad_moving_average_width
        self.vad_max_silence_length = vad_max_silence_length

        self.n_fft = int(mel_window_length * sampling_rate / 1000)
        self.hop_length = int(mel_window_step * sampling_rate / 1000)
        self.n_mels = n_mels

        self.partial_n_frames = partial_n_frames
        self.min_pad_coverage = min_pad_coverage
        self.partial_overlap_ratio = partial_overlap_ratio

    def preprocess_wav(self, fpath_or_wav, source_sr=None):
        # Load the wav from disk if needed
        if isinstance(fpath_or_wav, (str, Path)):
            wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
        else:
            wav = fpath_or_wav

        # Resample if numpy.array is passed and sr does not match
        if source_sr is not None and source_sr != self.sampling_rate:
            wav = librosa.resample(wav, source_sr, self.sampling_rate)

        # loudness normalization
        wav = normalize_volume(
            wav, self.audio_norm_target_dBFS, increase_only=True)

        # trim long silence
        if webrtcvad:
            wav = trim_long_silences(
                wav, self.vad_window_length, self.vad_moving_average_width,
                self.vad_max_silence_length, self.sampling_rate)
        return wav

    def melspectrogram(self, wav):
        mel = librosa.feature.melspectrogram(
            wav,
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels)
        mel = mel.astype(np.float32).T
        return mel

    def extract_mel_partials(self, wav):
        wav_slices, mel_slices = compute_partial_slices(
            len(wav), self.partial_n_frames, self.hop_length,
            self.min_pad_coverage, self.partial_overlap_ratio)

        # pad audio if needed
        max_wave_length = wav_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

        # Split the utterance into partials
        frames = self.melspectrogram(wav)
        frames_batch = np.array([frames[s] for s in mel_slices])
        return frames_batch  # [B, T, C]
