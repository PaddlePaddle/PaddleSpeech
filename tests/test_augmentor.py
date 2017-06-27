"""Test augmentor class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from data_utils import audio
from data_utils.augmentor.augmentation import AugmentationPipeline
import random
import numpy as np

random_seed = 0
#audio instance
audio_data = [3.0517571e-05, -8.54492188e-04, -1.09863281e-03, -9.4604492e-04,\
            -1.31225586e-03, -1.09863281e-03, -1.73950195e-03, -2.1057189e-03,\
            -2.04467773e-03, -1.46484375e-03, -1.43432617e-03, -9.4604492e-04,\
            -1.95312500e-03, -1.86157227e-03, -2.10571289e-03, -2.3193354e-03,\
            -2.01416016e-03, -2.62451172e-03, -2.07519531e-03, -2.3803719e-03]
audio_data = np.array(audio_data)
samplerate = 10


class TestAugmentor(unittest.TestCase):
    def test_volume(self):
        config_json = '[{"type": "volume","params": {"min_gain_dBFS": -15, '\
        '"max_gain_dBFS": 15},"prob": 1.0}]'
        aug_pipeline = AugmentationPipeline(
            augmentation_config=config_json, random_seed=random_seed)
        audio_seg = audio.AudioSegment(audio_data, samplerate)
        aug_pipeline.transform_audio(audio_seg)
        orig_audio = audio.AudioSegment(audio_data, samplerate)
        self.assertFalse(np.any(audio_seg.samples == orig_audio.samples))

    def test_speed(self):
        config_json = '[{"type":"speed","params": {"min_speed_rate": 0.9,' \
        '"max_speed_rate": 1.1},"prob": 1.0}]'
        aug_pipeline = AugmentationPipeline(
            augmentation_config=config_json, random_seed=random_seed)
        audio_seg = audio.AudioSegment(audio_data, samplerate)
        aug_pipeline.transform_audio(audio_seg)
        orig_audio = audio.AudioSegment(audio_data, samplerate)
        self.assertFalse(np.any(audio_seg.samples == orig_audio.samples))

    def test_resample(self):
        config_json = '[{"type":"resample","params": {"new_sample_rate":5},'\
        '"prob": 1.0}]'
        aug_pipeline = AugmentationPipeline(
            augmentation_config=config_json, random_seed=random_seed)
        audio_seg = audio.AudioSegment(audio_data, samplerate)
        aug_pipeline.transform_audio(audio_seg)
        self.assertTrue(audio_seg.sample_rate == 5)

    def test_bayesial(self):
        config_json = '[{"type":"bayesian_normal","params":{"target_db":-20,' \
        '"prior_db":-4, "prior_samples": -8, "startup_delay": 0.0},"prob":1.0}]'
        aug_pipeline = AugmentationPipeline(
            augmentation_config=config_json, random_seed=random_seed)
        audio_seg = audio.AudioSegment(audio_data, samplerate)
        aug_pipeline.transform_audio(audio_seg)
        orig_audio = audio.AudioSegment(audio_data, samplerate)
        self.assertFalse(np.any(audio_seg.samples == orig_audio.samples))


if __name__ == '__main__':
    unittest.main()
