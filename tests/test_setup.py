"""Test Setup."""
import unittest
import numpy as np
import os


class TestSetup(unittest.TestCase):
    def test_soundfile(self):
        import soundfile as sf
        # floating point data is typically limited to the interval [-1.0, 1.0],
        # but smaller/larger values are supported as well
        data = np.array([[1.75, -1.75], [1.0, -1.0], [0.5, -0.5],
                         [0.25, -0.25]])
        file = 'test.wav'
        sf.write(file, data, 44100, format='WAV', subtype='FLOAT')
        read, fs = sf.read(file)
        self.assertTrue(np.all(read == data))
        self.assertEqual(fs, 44100)
        os.remove(file)


if __name__ == '__main__':
    unittest.main()
