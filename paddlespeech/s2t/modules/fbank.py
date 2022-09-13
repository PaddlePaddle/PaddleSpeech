


import paddle
from paddle import nn

from paddlespeech.audio.compliance import kaldi

from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()

__all__ = ['KaldiFbank']

class KaldiFbank(nn.Layer):
    def __init__(self,
            fs=16000,
            n_mels=80,
            n_shift=160,  # unit:sample, 10ms
            win_length=400,  # unit:sample, 25ms
            energy_floor=0.0,
            dither=0.0):
        """
        Args:
            fs (int): sample rate of the audio
            n_mels (int): number of mel filter banks
            n_shift (int): number of points in a frame shift
            win_length (int): number of points in a frame windows
            energy_floor (float): Floor on energy in Spectrogram computation (absolute)
            dither (float): Dithering constant. Default 0.0
        """
        super().__init__()
        self.fs = fs
        self.n_mels = n_mels
        num_point_ms = fs / 1000
        self.n_frame_length = win_length / num_point_ms
        self.n_frame_shift = n_shift / num_point_ms
        self.energy_floor = energy_floor
        self.dither = dither

    def __repr__(self):
        return (
            "{name}(fs={fs}, n_mels={n_mels}, "
            "n_frame_shift={n_frame_shift}, n_frame_length={n_frame_length}, "
            "dither={dither}))".format(
                name=self.__class__.__name__,
                fs=self.fs,
                n_mels=self.n_mels,
                n_frame_shift=self.n_frame_shift,
                n_frame_length=self.n_frame_length,
                dither=self.dither, ))

    def forward(self, x: paddle.Tensor):
        """
        Args:
            x (paddle.Tensor): shape (Ti). 
                Not support: [Time, Channel] and Batch mode.

        Returns:
            paddle.Tensor: (T, D)
        """
        assert x.ndim == 1

        feat = kaldi.fbank(
            x.unsqueeze(0), # append channel dim, (C, Ti)
            n_mels=self.n_mels,
            frame_length=self.n_frame_length,
            frame_shift=self.n_frame_shift,
            dither=self.dither,
            energy_floor=self.energy_floor,
            sr=self.fs)

        assert feat.ndim == 2 # (T,D)
        return feat
