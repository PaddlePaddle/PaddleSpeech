#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright     2020    Zeng Xingui(zengxingui@baidu.com)
#
########################################################################

"""
Augment operator for audio
"""

import sys
import random
import numpy as np
import augly.audio as audaugs


class BaseAugment(object):
    """
    Base class of all augments for audio
    """
    def __init__(self):
        pass

    def __call__(self, inputs):
        return self._apply(inputs)

    def _apply(self, inputs):
        raise NotImplementedError

class AddNoise(BaseAugment):
    """
    Mixes in a background sound into the audio
    """
    def __init__(self, background_audio_list=None, min_snr_level=5.0, max_snr_level=20.0, seed=None):
        """
        @param background_audio_list: the list of path to the background audio, background audio will choose randomly
        from the list. If set to `None`, the background audio will be white noise
        @param snr_level_db: signal-to-noise ratio in dB
        @param seed: a NumPy random generator (or seed) such that the results remain reproducible
        """
        self._seed = seed
        self._background_audio_list = background_audio_list
        self._min_snr_level = min_snr_level
        self._max_snr_level = max_snr_level
        random.seed(seed)

    def _apply(self, inputs):
        """
        @param inputs: the path to the audio or a variable of type np.ndarray that will be augmented
        @returns: the augmented audio array
        """
        if self._min_snr_level == self._max_snr_level:
            snr_level_db = self._min_snr_level
        else:
            snr_level_db = random.uniform(self._min_snr_level, self._max_snr_level)
        if self._background_audio_list is None:
            background_audio = None
        else:
            lines = open(self._background_audio_list).read().splitlines()
            background_audio = random.choice(lines)
        output, sample = audaugs.add_background_noise(inputs, background_audio=background_audio,
                                                        snr_level_db=snr_level_db, seed=self._seed)
        return output

class ChangeVolume(BaseAugment):
    """
    Changes the volume of the audio
    """
    def __init__(self, min_volume_db=-1.25, max_volume_db=2.0, seed=None):
        """
        @param volume_db: the decibel amount by which to either increase (positive value) or decrease (negative value)
        the volume of the audio. If not provided, then the volume will be chosen randomly to be between
        min_volume_db and max_volume_db
        """
        random.seed(seed)
        self._min_volume_db = min_volume_db
        self._max_volume_db = max_volume_db

    def _apply(self, inputs):
        """
        @param inputs: the path to the audio or a variable of type np.ndarray that will be augmented
        @returns: the augmented audio array
        """
        if self._min_volume_db == self._max_volume_db:
            volume_db = self._min_volume_db
        else:
            volume_db = random.uniform(self._min_volume_db, self._max_volume_db)
        output, sample = audaugs.change_volume(inputs, volume_db=volume_db)
        return output

class Clicks(BaseAugment):
    """
    Adds clicks to the audio at a given regular interval
    """
    def __init__(self, min_seconds_between_clicks=0.2, max_seconds_between_clicks=1.0,
                 min_snr_level=1.0, max_snr_level=10.0, seed=None):
        """
        @param seconds_between_clicks: the amount of time between each click that will be added to the audio, in seconds
        @param snr_level_db: signal-to-noise ratio in dB
        """
        self._min_seconds_between_clicks = min_seconds_between_clicks
        self._max_seconds_between_clicks = max_seconds_between_clicks
        self._min_snr_level = min_snr_level
        self._max_snr_level = max_snr_level
        random.seed(seed)

    def _apply(self, inputs):
        """
        @param inputs: the path to the audio or a variable of type np.ndarray that will be augmented
        @returns: the augmented audio array
        """
        if self._min_seconds_between_clicks == self._max_seconds_between_clicks:
            seconds_between_clicks = self._min_seconds_between_clicks
        else:
            seconds_between_clicks = random.uniform(self._min_seconds_between_clicks, self._max_seconds_between_clicks)
        if self._min_snr_level == self._max_snr_level:
            snr_level_db = self._min_snr_level
        else:
            snr_level_db = random.uniform(self._min_snr_level, self._max_snr_level)
        output, sample = audaugs.clicks(inputs, seconds_between_clicks=seconds_between_clicks,
                                        snr_level_db=snr_level_db)
        return output

class Clip(BaseAugment):
    """
    Clips the audio using the specified offset and duration factors
    """
    def __init__(self, min_offset_factor=0.0, max_offset_factor=1.0, min_duration_factor=0.0, max_duration_factor=1.0,
                 seed=None):
        """
        @param offset_factor: start point of the crop relative to the audio duration (this parameter is multiplied by
        the audio duration)
        @param duration_factor: the length of the crop relative to the audio duration (this parameter is multiplied by
        the audio duration)
        """
        self._min_offset_factor = min_offset_factor
        self._max_offset_factor = max_offset_factor
        self._min_duration_factor = min_duration_factor
        self._max_duration_factor = max_duration_factor
        random.seed(seed)

    def _apply(self, inputs):
        """
        @param inputs: the path to the audio or a variable of type np.ndarray that will be augmented
        @returns: the augmented audio array
        """
        if self._min_offset_factor == self._max_offset_factor:
            offset_factor = self._min_offset_factor
        else:
            offset_factor = random.uniform(self._min_offset_factor, self._max_offset_factor)
        if self._min_duration_factor == self._max_duration_factor:
            duration_factor = self._min_duration_factor
        else:
            duration_factor = random.uniform(self._min_duration_factor, 1.0 - offset_factor)
        output, sample = audaugs.clip(inputs, offset_factor=offset_factor, duration_factor=duration_factor)
        return output

class Normalize(BaseAugment):
    """
    Normalizes the audio array along the chosen axis (norm(audio, axis=axis) == 1)
    """
    def __init__(self, norm=np.inf, axis=0, threshold=None, fill=None):
        """
        @param norm: the type of norm to compute:
            - np.inf: maximum absolute value
            - -np.inf: mininum absolute value
            - 0: number of non-zeros (the support)
            - float: corresponding l_p norm
            - None: no normalization is performed
        @param axis: axis along which to compute the norm
        @param threshold: if provided, only the columns (or rows) with norm of at least `threshold` are normalized
        @param fill: if None, then columns (or rows) with norm below `threshold` are left as is. If False, then columns
        (rows) with norm below `threshold` are set to 0. If True, then columns (rows) with norm below `threshold` are
        filled uniformly such that the corresponding norm is 1
        """
        self._norm = norm
        self._axis = axis
        self._threshold = threshold
        self._fill = fill

    def _apply(self, inputs):
        """
        @param inputs: the path to the audio or a variable of type np.ndarray that will be augmented
        @returns: the augmented audio array
        """
        output, sample = audaugs.normalize(inputs, norm=self._norm, axis=self._axis, fill=self._fill)
        return output

class Reverb(BaseAugment):
    """
    Adds reverberation to the audio
    """
    def __init__(self, room_type=None, stereo_depth=0.0, wet_gain=0.0, wet_only=False, seed=None):
        """
        @param reverberance: (%) sets the length of the reverberation tail. This determines how long the reverberation
        continues for after the original sound being reverbed comes to an end, and so simulates the "liveliness" of the
        room acoustics
        @param hf_damping: (%) increasing the damping produces a more "muted" effect. The reverberation does not build
        up as much, and the high frequencies decay faster than the low frequencies
        @param room_scale: (%) sets the size of the simulated room. A high value will simulate the reverberation effect
        of a large room and a low value will simulate the effect of a small room
        @param stereo_depth: (%) sets the apparent "width" of the reverb effect for stereo tracks only. Increasing this
        value applies more variation between left and right channels, creating a more "spacious" effect. When set at
        zero, the effect is applied independently to left and right channels
        @param pre_delay: (ms) delays the onset of the reverberation for the set time after the start of the original
        input. This also delays the onset of the reverb tail
        @param wet_gain: (db) applies volume adjustment to the reverberation ("wet") component in the mix
        @param wet_only: only the wet signal (added reverberation) will be in the resulting output, and the original
        audio will be removed
        """
        random.seed(seed)
        self._room_type = room_type
        self._stereo_depth = stereo_depth
        self._wet_gain = wet_gain
        self._wet_only = wet_only

    def _apply(self, inputs):
        """
        @param inputs: the path to the audio or a variable of type np.ndarray that will be augmented
        @returns: the augmented audio array
        """
        if self._room_type is None:
            room_type = random.choice(["small", "medium", "large"])
        if room_type is "small":
            room_scale = random.uniform(0.0, 30.0)
            reverberance = random.uniform(10.0, 40.0)
            pre_delay = random.uniform(0.0, 15.0)
            hf_damping = random.uniform(10.0, 60.0)
        elif room_type is "medium":
            room_scale = random.uniform(30.0, 70.0)
            reverberance = random.uniform(30.0, 60.0)
            pre_delay = random.uniform(0.0, 20.0)
            hf_damping = random.uniform(20.0, 70.0)
        else:
            room_scale = random.uniform(70.0, 100.0)
            reverberance = random.uniform(50.0, 70.0)
            pre_delay = random.uniform(0.0, 30.0)
            hf_damping = random.uniform(30.0, 70.0)
        output, sample = audaugs.reverb(inputs, reverberance=reverberance, hf_damping=hf_damping,
                                        room_scale=room_scale, stereo_depth=self._stereo_depth,
                                        pre_delay=pre_delay, wet_gain=self._wet_gain, wet_only=self._wet_only)
        return output

class Tempo(BaseAugment):
    """
    Adjusts the tempo of the audio by a given factor
    """
    def __init__(self, min_factor=0.85, max_factor=1.25, seed=None):
        """
        @param factor: the tempo factor. If rate > 1 the audio will be sped up by that factor; if rate < 1 the audio
        will be slowed down by that factor, without affecting the pitch. If not provided, then the rate will be chosen
        randomly to be between min_factor and max_factor
        """
        random.seed(seed)
        self._min_factor = min_factor
        self._max_factor = max_factor

    def _apply(self, inputs):
        """
        @param inputs: the path to the audio or a variable of type np.ndarray that will be augmented
        @returns: the augmented audio array
        """
        if self._min_factor == self._max_factor:
            factor = self._min_factor
        else:
            factor = random.uniform(self._min_factor, self._max_factor)
        output, sample = audaugs.tempo(inputs, factor=factor)
        return output



if __name__ == "__main__":
    a = np.array([1, 2, 3, 4, 5, 6])

    t = AddNoise(background_audio_list="./noise_wav_list", seed=0)
    print("add noise {}".format(t(a)))

    t = Reverb(stereo_depth=100.0)
    print("stereo reverb {}".format(t(a)))

    t = Reverb()
    print("reverb {}".format(t(a)))

    t = Tempo(seed=0)
    print("speed perturb {}".format(t(a)))

    t = Clip()
    print("clip {}".format(t(a)))

    t = Clicks()
    print("clicks {}".format(t(a)))

    t = ChangeVolume()
    print("volume perturb {}".format(t(a)))

    t = Normalize()
    print("normalize {}".format(t(a)))
