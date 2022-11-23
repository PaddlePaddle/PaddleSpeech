from python_speech_features import sigproc
import unittest
import numpy as np
import time


class test_case(unittest.TestCase):
    def test_frame_sig(self):
        n = 10000124
        frame_len = 37
        frame_step = 13
        x = np.random.rand(n)
        t0 = time.time()
        y_old = sigproc.framesig(x, frame_len=frame_len, frame_step=frame_step, stride_trick=False)
        t1 = time.time()
        y_new = sigproc.framesig(x, frame_len=frame_len, frame_step=frame_step, stride_trick=True)
        t_new = time.time() - t1
        t_old = t1 - t0
        self.assertTupleEqual(y_old.shape, y_new.shape)
        np.testing.assert_array_equal(y_old, y_new)
        self.assertLess(t_new, t_old)
        print('new run time %3.2f < %3.2f sec' % (t_new, t_old))

    def test_rolling(self):
        x = np.arange(10)
        y = sigproc.rolling_window(x, window=4, step=3)
        y_expected = np.array([[0, 1, 2, 3],
                               [3, 4, 5, 6],
                               [6, 7, 8, 9]]
                              )
        y = np.testing.assert_array_equal(y, y_expected)
