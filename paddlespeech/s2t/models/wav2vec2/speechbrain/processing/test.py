import paddle
import numpy as np

def blackman_window(window_length, periodic=True):
    if window_length == 0:
        return []
    if window_length == 1:
        return paddle.ones([1])
    if periodic:
        window_length += 1
    
    window = paddle.arange(window_length) * (np.pi / (window_length - 1))
    window = 0.08 * paddle.cos(window * 4) - 0.5 * paddle.cos(window * 2) + 0.42
    return window[:-1] if periodic else window
