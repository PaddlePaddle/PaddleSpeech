import numpy as np


def delta(feat, window):
    assert window > 0
    delta_feat = np.zeros_like(feat)
    for i in range(1, window + 1):
        delta_feat[:-i] += i * feat[i:]
        delta_feat[i:] += -i * feat[:-i]
        delta_feat[-i:] += i * feat[-1]
        delta_feat[:i] += -i * feat[0]
    delta_feat /= 2 * sum(i ** 2 for i in range(1, window + 1))
    return delta_feat


def add_deltas(x, window=2, order=2):
    """
    Args:
        x (np.ndarray): speech feat, (T, D).

    Return:
        np.ndarray: (T, (1+order)*D)
    """
    feats = [x]
    for _ in range(order):
        feats.append(delta(feats[-1], window))
    return np.concatenate(feats, axis=1)


class AddDeltas():
    def __init__(self, window=2, order=2):
        self.window = window
        self.order = order

    def __repr__(self):
        return "{name}(window={window}, order={order}".format(
            name=self.__class__.__name__, window=self.window, order=self.order
        )

    def __call__(self, x):
        return add_deltas(x, window=self.window, order=self.order)
