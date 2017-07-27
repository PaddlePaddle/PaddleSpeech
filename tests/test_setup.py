"""Test Setup."""
import unittest


class TestSetup(unittest.TestCase):
    # test the installation of libsndfile library
    def test_soundfile(self):
        import soundfile


if __name__ == '__main__':
    unittest.main()
