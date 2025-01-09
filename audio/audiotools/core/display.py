# MIT License, Copyright (c) 2023-Present, Descript.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Modified from audiotools(https://github.com/descriptinc/audiotools/blob/master/audiotools/core/display.py)
import inspect
import typing
from functools import wraps

from . import util


def format_figure(func):
    """Decorator for formatting figures produced by the code below.
    See :py:func:`audiotools.core.util.format_figure` for more.

    Parameters
    ----------
    func : Callable
        Plotting function that is decorated by this function.

    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        f_keys = inspect.signature(util.format_figure).parameters.keys()
        f_kwargs = {}
        for k, v in list(kwargs.items()):
            if k in f_keys:
                kwargs.pop(k)
                f_kwargs[k] = v
        func(*args, **kwargs)
        util.format_figure(**f_kwargs)

    return wrapper


class DisplayMixin:
    @format_figure
    def specshow(
            self,
            preemphasis: bool=False,
            x_axis: str="time",
            y_axis: str="linear",
            n_mels: int=128,
            **kwargs, ):
        """Displays a spectrogram, using ``librosa.display.specshow``.

        Parameters
        ----------
        preemphasis : bool, optional
            Whether or not to apply preemphasis, which makes high
            frequency detail easier to see, by default False
        x_axis : str, optional
            How to label the x axis, by default "time"
        y_axis : str, optional
            How to label the y axis, by default "linear"
        n_mels : int, optional
            If displaying a mel spectrogram with ``y_axis = "mel"``,
            this controls the number of mels, by default 128.
        kwargs : dict, optional
            Keyword arguments to :py:func:`audiotools.core.util.format_figure`.
        """
        import librosa
        import librosa.display

        # Always re-compute the STFT data before showing it, in case
        # it changed.
        signal = self.clone()
        signal.stft_data = None

        if preemphasis:
            signal.preemphasis()

        ref = signal.magnitude.max()
        log_mag = signal.log_magnitude(ref_value=ref)

        if y_axis == "mel":
            log_mag = 20 * signal.mel_spectrogram(n_mels).clip(1e-5).log10()
            log_mag -= log_mag.max()

        librosa.display.specshow(
            log_mag.numpy()[0].mean(axis=0),
            x_axis=x_axis,
            y_axis=y_axis,
            sr=signal.sample_rate,
            **kwargs, )

    @format_figure
    def waveplot(self, x_axis: str="time", **kwargs):
        """Displays a waveform plot, using ``librosa.display.waveshow``.

        Parameters
        ----------
        x_axis : str, optional
            How to label the x axis, by default "time"
        kwargs : dict, optional
            Keyword arguments to :py:func:`audiotools.core.util.format_figure`.
        """
        import librosa
        import librosa.display

        audio_data = self.audio_data[0].mean(axis=0)
        audio_data = audio_data.cpu().numpy()

        plot_fn = "waveshow" if hasattr(librosa.display,
                                        "waveshow") else "waveplot"
        wave_plot_fn = getattr(librosa.display, plot_fn)
        wave_plot_fn(audio_data, x_axis=x_axis, sr=self.sample_rate, **kwargs)

    @format_figure
    def wavespec(self, x_axis: str="time", **kwargs):
        """Displays a waveform plot, using ``librosa.display.waveshow``.

        Parameters
        ----------
        x_axis : str, optional
            How to label the x axis, by default "time"
        kwargs : dict, optional
            Keyword arguments to :py:func:`audiotools.core.display.DisplayMixin.specshow`.
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        gs = GridSpec(6, 1)
        plt.subplot(gs[0, :])
        self.waveplot(x_axis=x_axis)
        plt.subplot(gs[1:, :])
        self.specshow(x_axis=x_axis, **kwargs)

    def write_audio_to_tb(
            self,
            tag: str,
            writer,
            step: int=None,
            plot_fn: typing.Union[typing.Callable, str]="specshow",
            **kwargs, ):
        """Writes a signal and its spectrogram to Tensorboard. Will show up
        under the Audio and Images tab in Tensorboard.

        Parameters
        ----------
        tag : str
            Tag to write signal to (e.g. ``clean/sample_0.wav``). The image will be
            written to the corresponding ``.png`` file (e.g. ``clean/sample_0.png``).
        writer : SummaryWriter
            A SummaryWriter object from PyTorch library.
        step : int, optional
            The step to write the signal to, by default None
        plot_fn : typing.Union[typing.Callable, str], optional
            How to create the image. Set to ``None`` to avoid plotting, by default "specshow"
        kwargs : dict, optional
            Keyword arguments to :py:func:`audiotools.core.display.DisplayMixin.specshow` or
            whatever ``plot_fn`` is set to.
        """
        import matplotlib.pyplot as plt

        audio_data = self.audio_data[0, 0].detach().cpu().numpy()
        sample_rate = self.sample_rate
        writer.add_audio(tag, audio_data, step, sample_rate)

        if plot_fn is not None:
            if isinstance(plot_fn, str):
                plot_fn = getattr(self, plot_fn)
            fig = plt.figure()
            plt.clf()
            plot_fn(**kwargs)
            writer.add_figure(tag.replace("wav", "png"), fig, step)

    def save_image(
            self,
            image_path: str,
            plot_fn: typing.Union[typing.Callable, str]="specshow",
            **kwargs, ):
        """Save AudioSignal spectrogram (or whatever ``plot_fn`` is set to) to
        a specified file.

        Parameters
        ----------
        image_path : str
            Where to save the file to.
        plot_fn : typing.Union[typing.Callable, str], optional
            How to create the image. Set to ``None`` to avoid plotting, by default "specshow"
        kwargs : dict, optional
            Keyword arguments to :py:func:`audiotools.core.display.DisplayMixin.specshow` or
            whatever ``plot_fn`` is set to.
        """
        import matplotlib.pyplot as plt

        if isinstance(plot_fn, str):
            plot_fn = getattr(self, plot_fn)

        plt.clf()
        plot_fn(**kwargs)
        plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
        plt.close()
