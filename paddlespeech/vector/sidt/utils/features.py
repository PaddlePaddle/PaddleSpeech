#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright     2021    Zeng Xingui(zengxingui@baidu.com)
#
########################################################################

"""
extract audio features
"""
import os
import sys
import random
import paddle
import numpy as np
import soundfile as sf
# import torchaudio
# import torch

from sidt import _logger as log
from sidt.utils.utils import is_exist


def load_audio(audio_file, start_time=0, duration=-1, target_sample_rate=8000, random_chunk=True):
    """
    Load audio segment

    Args:
        audio_file: audio file path
        start_time: start time
        duration: segment duration
        sample_rate: sample rate of the audio
        random_chunk: read a segment randomly

    Returns:
        numpy.array.
    """
    # FixMe. Currently only support mono channel, bitwidth=16
    if not is_exist(audio_file):
        raise

    # get audio meta
    audio_meta = sf.info(audio_file)
    sample_rate = audio_meta.sample_rate
    assert audio_meta.num_channels == 1
    assert audio_meta.bits_per_sample == 16
    num_frames = audio_meta.num_frames if duration == -1 else int(duration * sample_rate)

    if random_chunk and num_frames < audio_meta.num_frames:
        start_time = random.randint(0, audio_meta.num_frames - num_frames)
    else:
        start_time = int(start_time * sample_rate)

    # read audio file
    sig, fs = sf.read(audio_file, start=start_time,
                             frames=num_frames)

    # apply codec
    # if audio_meta.sample_rate != target_sample_rate:
    #     sig = torchaudio.compliance.kaldi.resample_waveform(sig, fs, target_sample_rate)

    return sig.numpy()


def save_audio(audio_file, sig, sample_rate):
    """ save audio
    """
    sf.write(audio_file, sig, sample_rate=sample_rate,)
    # torchaudio.save(audio_file, torch.Tensor(sig), sample_rate=sample_rate, format="wav", encoding="PCM_S")


def vad(feats, energy_mean_scale=0.5, energy_threshold=5, frame_context=0, proportion_threshold=0.4):
    """ Kaldi energy based Voice activity detection
    """
    assert energy_mean_scale > 0.0
    log_energy = feats[0, :]
    delta =  np.sum(log_energy) * energy_mean_scale / feats.shape[1]
    energy_threshold += np.sum(log_energy) * energy_mean_scale / feats.shape[1]

    vad_val = np.zeros(feats.shape[1], dtype=np.bool)
    for idx in range(feats.shape[1]):
        num_count = 0
        den_count = 0
        end = idx + min(feats.shape[1], frame_context + 1)
        bg = max(0, idx - frame_context)
        den_count = end - bg
        num_count = np.sum(log_energy[bg: end] > energy_threshold)
        vad_val[idx] = num_count >= den_count * proportion_threshold

    vad_feats = feats[:, vad_val]
    return vad_feats


def fbank(sig, dim, sample_rate=8000):
    """ compute fbank
    """
    return torchaudio.compliance.kaldi.fbank(torch.Tensor(sig), num_mel_bins=dim, sample_frequency=sample_rate).numpy()


def mfcc(sig, num_mel_bins=23, num_ceps=13, sample_rate=8000):
    """ compute mfcc
    """
    return torchaudio.compliance.kaldi.mfcc(torch.Tensor(sig), num_mel_bins=num_mel_bins, num_ceps=num_ceps,
                                            sample_frequency=sample_rate).numpy()


def sliding_window_cmvn(feats, cmvn_window=300, center=True, norm_vars=False):
    """ apply cmvn
    """
    return torchaudio.functional.sliding_window_cmn(torch.Tensor(feats), cmn_window=cmvn_window, center=center,
                                                    norm_vars=norm_vars).numpy()


if __name__ == "__main__":
    sample_rate = 8000
    sig = load_audio(sys.argv[1], 0, -1, sample_rate, False)
    save_audio("audio.wav", sig, sample_rate)
    fbank_feats = fbank(sig, 80, sample_rate)
    print(fbank_feats.shape)
    fbank_feats = mfcc(sig, sample_rate=sample_rate)
    print(fbank_feats.shape)
    cmvn_feats = sliding_window_cmvn(fbank_feats)
    vad_feats = vad(cmvn_feats.transpose())
    print(fbank_feats, fbank_feats.shape)
    print(cmvn_feats, cmvn_feats.shape)
    print(vad_feats, vad_feats.shape)
