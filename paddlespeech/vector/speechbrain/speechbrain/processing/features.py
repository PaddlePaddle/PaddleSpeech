"""Low-level feature pipeline components

This library gathers functions that compute popular speech  features over
batches of data. All the classes are of type nn.Layer. This gives the
possibility to have end-to-end  differentiability and to backpropagate the
gradient through them. Our functions are a modified version the ones
in paddle audio toolkit (https://github.com/pytorch/audio).

Example
-------
>>> import paddle
>>> from speechbrain.dataio.dataio import read_audio
>>> signal =read_audio('samples/audio_samples/example1.wav')
>>> signal = signal.unsqueeze(0)
>>> compute_STFT = STFT(
...     sample_rate=16000, win_length=25, hop_length=10, n_fft=400
... )
>>> features = compute_STFT(signal)
>>> features = spectral_magnitude(features)
>>> compute_fbanks = Filterbank(n_mels=40)
>>> features = compute_fbanks(features)
>>> compute_mfccs = DCT(input_size=40, n_out=20)
>>> features = compute_mfccs(features)
>>> compute_deltas = Deltas(input_size=20)
>>> delta1 = compute_deltas(features)
>>> delta2 = compute_deltas(delta1)
>>> features = paddle.cat([features, delta1, delta2], dim=2)
>>> compute_cw = ContextWindow(left_frames=5, right_frames=5)
>>> features  = compute_cw(features)
>>> norm = InputNormalization()
>>> features = norm(features, paddle.tensor([1]).float())

Authors
 * Mirco Ravanelli 2020
"""
import math
import paddle
import logging
import numpy as np
from packaging import version
from speechbrain.utils.checkpoints import (
    mark_as_saver,
    mark_as_loader,
    mark_as_transfer,
    register_checkpoint_hooks,
)

# logger = logging.getLogger(__name__)
from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()

class STFT(paddle.nn.Layer):
    """computes the Short-Term Fourier Transform (STFT).

    This class computes the Short-Term Fourier Transform of an audio signal.
    It supports multi-channel audio inputs (batch, time, channels).

    Arguments
    ---------
    sample_rate : int
        Sample rate of the input audio signal (e.g 16000).
    win_length : float
        Length (in ms) of the sliding window used to compute the STFT.
    hop_length : float
        Length (in ms) of the hope of the sliding window used to compute
        the STFT.
    n_fft : int
        Number of fft point of the STFT. It defines the frequency resolution
        (n_fft should be <= than win_len).
    window_fn : function
        A function that takes an integer (number of samples) and outputs a
        tensor to be multiplied with each window before fft.
    normalized_stft : bool
        If True, the function returns the  normalized STFT results,
        i.e., multiplied by win_length^-0.5 (default is False).
    center : bool
        If True (default), the input will be padded on both sides so that the
        t-th frame is centered at time t×hop_length. Otherwise, the t-th frame
        begins at time t×hop_length.
    pad_mode : str
        It can be 'constant','reflect','replicate', 'circular', 'reflect'
        (default). 'constant' pads the input tensor boundaries with a
        constant value. 'reflect' pads the input tensor using the reflection
        of the input boundary. 'replicate' pads the input tensor using
        replication of the input boundary. 'circular' pads using  circular
        replication.
    onesided : True
        If True (default) only returns nfft/2 values. Note that the other
        samples are redundant due to the Fourier transform conjugate symmetry.

    Example
    -------
    >>> import paddle
    >>> compute_STFT = STFT(
    ...     sample_rate=16000, win_length=25, hop_length=10, n_fft=400
    ... )
    >>> inputs = paddle.randn([10, 16000])
    >>> features = compute_STFT(inputs)
    >>> features.shape
    paddle.Size([10, 101, 201, 2])
    """

    def __init__(
        self,
        sample_rate,
        win_length=25,
        hop_length=10,
        n_fft=400,
        window_fn=np.hamming,
        normalized_stft=False,
        center=True,
        pad_mode="constant",
        onesided=True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.normalized_stft = normalized_stft
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided

        # Convert win_length and hop_length from ms to samples
        self.win_length = int(
            round((self.sample_rate / 1000.0) * self.win_length)
        )
        self.hop_length = int(
            round((self.sample_rate / 1000.0) * self.hop_length)
        )

        self.window = paddle.to_tensor(window_fn(self.win_length), dtype="float32")

    def forward(self, x):
        """Returns the STFT generated from the input waveforms.

        Arguments
        ---------
        x : tensor
            A batch of audio signals to transform.
        """

        # Managing multi-channel stft
        or_shape = x.shape
        if len(or_shape) == 3:
            x = x.transpose(perm=[0, 2, 1])
            x = x.reshape([or_shape[0] * or_shape[2], or_shape[1]])
        stft = paddle.signal.stft(
                x,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                self.center,
                self.pad_mode,
                self.normalized_stft,
                self.onesided,
            )

        # Retrieving the original dimensionality (batch,time, channels)
        if len(or_shape) == 3:
            stft = stft.reshape(
                [or_shape[0],
                or_shape[2],
                stft.shape[1],
                stft.shape[2]],
            )
            # paddle 计算出来的结果是复数，torch计算出来的结果是两个实数表示的一个复数
            stft = stft.transpose(perm=[0, 3, 2, 1])
        else:
            # (batch, time, channels)
            stft = stft.transpose(perm=[0, 2, 1])

        return stft


class ISTFT(paddle.nn.Layer):
    """ Computes the Inverse Short-Term Fourier Transform (ISTFT)

    This class computes the Inverse Short-Term Fourier Transform of
    an audio signal. It supports multi-channel audio inputs
    (batch, time_step, n_fft, 2, n_channels [optional]).

    Arguments
    ---------
    sample_rate : int
        Sample rate of the input audio signal (e.g. 16000).
    win_length : float
        Length (in ms) of the sliding window used when computing the STFT.
    hop_length : float
        Length (in ms) of the hope of the sliding window used when computing
        the STFT.
    window_fn : function
        A function that takes an integer (number of samples) and outputs a
        tensor to be used as a window for ifft.
    normalized_stft : bool
        If True, the function assumes that it's working with the normalized
        STFT results. (default is False)
    center : bool
        If True (default), the function assumes that the STFT result was padded
        on both sides.
    onesided : True
        If True (default), the function assumes that there are n_fft/2 values
        for each time frame of the STFT.
    epsilon : float
        A small value to avoid division by 0 when normalizing by the sum of the
        squared window. Playing with it can fix some abnormalities at the
        beginning and at the end of the reconstructed signal. The default value
        of epsilon is 1e-12.

    Example
    -------
    >>> import paddle
    >>> compute_STFT = STFT(
    ...     sample_rate=16000, win_length=25, hop_length=10, n_fft=400
    ... )
    >>> compute_ISTFT = ISTFT(
    ...     sample_rate=16000, win_length=25, hop_length=10
    ... )
    >>> inputs = paddle.randn([10, 16000])
    >>> outputs = compute_ISTFT(compute_STFT(inputs))
    >>> outputs.shape
    paddle.Size([10, 16000])
    """

    def __init__(
        self,
        sample_rate,
        n_fft=None,
        win_length=25,
        hop_length=10,
        window_fn=np.hamming,
        normalized_stft=False,
        center=True,
        onesided=True,
        epsilon=1e-12,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.normalized_stft = normalized_stft
        self.center = center
        self.onesided = onesided
        self.epsilon = epsilon

        # Convert win_length and hop_length from ms to samples
        self.win_length = int(
            round((self.sample_rate / 1000.0) * self.win_length)
        )
        self.hop_length = int(
            round((self.sample_rate / 1000.0) * self.hop_length)
        )

        # Create window using provided function
        self.window = paddle.to_tensor(window_fn(self.win_length), dtype="float32")

    def forward(self, x, sig_length=None):
        """ Returns the ISTFT generated from the input signal.

        Arguments
        ---------
        x : tensor
            A batch of audio signals in the frequency domain to transform.
        sig_length : int
            The length of the output signal in number of samples. If not
            specified will be equal to: (time_step - 1) * hop_length + n_fft
        """

        or_shape = x.shape

        # Infer n_fft if not provided
        if self.n_fft is None and self.onesided:
            n_fft = (x.shape[2] - 1) * 2
        elif self.n_fft is None and not self.onesided:
            n_fft = x.shape[2]
        else:
            n_fft = self.n_fft

        # Changing the format for (batch, time_step, n_fft, 2, n_channels)
        if len(or_shape) == 4:
            x = x.transpose([0, 3, 2, 1])

            # Lumping batch and channel dimension, because paddle.istft
            # doesn't support batching.
            x = x.reshape([-1, x.shape[2], x.shape[3]])
        elif len(or_shape) == 3:
            x = x.transpose([0, 2, 1])

        # isft ask complex input
        # x = paddle.complex(x[..., 0], x[..., 1])

        istft = paddle.signal.istft(
            x,
            n_fft=n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            onesided=self.onesided,
            length=sig_length,
        )

        # Convert back to (time, time_step, n_channels)
        if len(or_shape) == 4:
            istft = istft.reshape([or_shape[0], or_shape[3], -1])
            istft = istft.transpose(perm=[0, 2, 1])
        return istft


def spectral_magnitude(stft, power=1, log=False, eps=1e-14):
    """Returns the magnitude of a complex spectrogram.

    Arguments
    ---------
    stft : paddle.Tensor
        A tensor, output from the stft function.
    power : int
        What power to use in computing the magnitude.
        Use power=1 for the power spectrogram.
        Use power=0.5 for the magnitude spectrogram.
    log : bool
        Whether to apply log to the spectral features.

    Example
    -------
    >>> a = paddle.Tensor([[3, 4]])
    >>> spectral_magnitude(a, power=0.5)
    tensor([5.])
    """
    # logger.info("stft value: {}".format(stft.shape))
    # stft = [batch, time, dim]
    # stft_real_power = [batch, time, dim_real]
    stft_real_power = stft.real().pow(2).unsqueeze(-2)
    # stft_imag_power = [batch, time, dim_image]
    stft_imag_power = stft.imag().pow(2).unsqueeze(-2)
    # spectr = stft.pow(2).sum(-1)
    # concat <-([batch, time, 1, dim_real] [batch, time, 1, dim_real/dim_imag])
    # [batch, time, 2, dim]-> transpose  [batch, time, dim, 2] -> sum(axis=-1) -> [batch, time, dim]
    spectr = paddle.concat(
            [stft_real_power, stft_imag_power], axis=-2).transpose(
            perm=[0, 1, 3, 2]).sum(-1)
    # logger.info("spectr value: {}".format(spectr.shape))
    # Add eps avoids NaN when spectr is zero
    if power < 1:
        spectr = spectr + eps
    spectr = spectr.pow(power)

    if log:
        return paddle.log(spectr + eps)
    return spectr


class Filterbank(paddle.nn.Layer):
    """computes filter bank (FBANK) features given spectral magnitudes.

    Arguments
    ---------
    n_mels : float
        Number of Mel filters used to average the spectrogram.
    log_mel : bool
        If True, it computes the log of the FBANKs.
    filter_shape : str
        Shape of the filters ('triangular', 'rectangular', 'gaussian').
    f_min : int
        Lowest frequency for the Mel filters.
    f_max : int
        Highest frequency for the Mel filters.
    n_fft : int
        Number of fft points of the STFT. It defines the frequency resolution
        (n_fft should be<= than win_len).
    sample_rate : int
        Sample rate of the input audio signal (e.g, 16000)
    power_spectrogram : float
        Exponent used for spectrogram computation.
    amin : float
        Minimum amplitude (used for numerical stability).
    ref_value : float
        Reference value used for the dB scale.
    top_db : float
        Minimum negative cut-off in decibels.
    freeze : bool
        If False, it the central frequency and the band of each filter are
        added into nn.parameters. If True, the standard frozen features
        are computed.
    param_change_factor: bool
        If freeze=False, this parameter affects the speed at which the filter
        parameters (i.e., central_freqs and bands) can be changed.  When high
        (e.g., param_change_factor=1) the filters change a lot during training.
        When low (e.g. param_change_factor=0.1) the filter parameters are more
        stable during training
    param_rand_factor: float
        This parameter can be used to randomly change the filter parameters
        (i.e, central frequencies and bands) during training.  It is thus a
        sort of regularization. param_rand_factor=0 does not affect, while
        param_rand_factor=0.15 allows random variations within +-15% of the
        standard values of the filter parameters (e.g., if the central freq
        is 100 Hz, we can randomly change it from 85 Hz to 115 Hz).

    Example
    -------
    >>> import paddle
    >>> compute_fbanks = Filterbank()
    >>> inputs = paddle.randn([10, 101, 201])
    >>> features = compute_fbanks(inputs)
    >>> features.shape
    paddle.Size([10, 101, 40])
    """

    def __init__(
        self,
        n_mels=40,
        log_mel=True,
        filter_shape="triangular",
        f_min=0,
        f_max=8000,
        n_fft=400,
        sample_rate=16000,
        power_spectrogram=2,
        amin=1e-10,
        ref_value=1.0,
        top_db=80.0,
        param_change_factor=1.0,
        param_rand_factor=0.0,
        freeze=True,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.log_mel = log_mel
        self.filter_shape = filter_shape
        self.f_min = f_min
        self.f_max = f_max
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.power_spectrogram = power_spectrogram
        self.amin = amin
        self.ref_value = ref_value
        self.top_db = top_db
        self.freeze = freeze
        self.n_stft = self.n_fft // 2 + 1
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))
        self.device_inp = "cpu"
        self.param_change_factor = param_change_factor
        self.param_rand_factor = param_rand_factor

        if self.power_spectrogram == 2:
            self.multiplier = 10
        else:
            self.multiplier = 20

        # Make sure f_min < f_max
        if self.f_min >= self.f_max:
            err_msg = "Require f_min: %f < f_max: %f" % (
                self.f_min,
                self.f_max,
            )
            logger.error(err_msg, exc_info=True)

        # Filter definition
        mel = paddle.linspace(
            self._to_mel(self.f_min), self._to_mel(self.f_max), self.n_mels + 2,
            dtype="float32"
        )
        hz = self._to_hz(mel)

        # Computation of the filter bands
        band = hz[1:] - hz[:-1]
        self.band = band[:-1]
        self.f_central = hz[1:-1]

        # Adding the central frequency and the band to the list of nn param
        # todo 需要配置self.freeze参数
        # if not self.freeze:
        #     self.f_central = paddle.nn.Parameter(
        #         self.f_central / (self.sample_rate * self.param_change_factor)
        #     )
        #     self.band = paddle.nn.Parameter(
        #         self.band / (self.sample_rate * self.param_change_factor)
        #     )

        # Frequency axis
        all_freqs = paddle.linspace(0, self.sample_rate // 2, self.n_stft, dtype="float32")

        # Replicating for all the filters
        self.all_freqs_mat = all_freqs.tile([self.f_central.shape[0], 1])

    def forward(self, spectrogram):
        """Returns the FBANks.

        Arguments
        ---------
        x : tensor
            A batch of spectrogram tensors.
        """
        # Computing central frequency and bandwidth of each filter
        f_central_mat = self.f_central.tile(
            [self.all_freqs_mat.shape[1], 1]
            ).transpose(perm=[1, 0])
        # print("f_central_mat shape: {} ".format(f_central_mat.shape))
        band_mat = self.band.tile([self.all_freqs_mat.shape[1], 1]
            ).transpose(perm=[1, 0])
        # print("band_mat shape: {} ".format(band_mat.shape))
        # Uncomment to print filter parameters
        # print(self.f_central*self.sample_rate * self.param_change_factor)
        # print(self.band*self.sample_rate* self.param_change_factor)

        # Creation of the multiplication matrix. It is used to create
        # the filters that average the computed spectrogram.
        if not self.freeze:
            f_central_mat = f_central_mat * (
                self.sample_rate
                * self.param_change_factor
                * self.param_change_factor
            )
            band_mat = band_mat * (
                self.sample_rate
                * self.param_change_factor
                * self.param_change_factor
            )

        # Regularization with random changes of filter central frequency and band
        elif self.param_rand_factor != 0 and self.training:
            rand_change = (
                1.0
                + paddle.rand(2) * 2 * self.param_rand_factor
                - self.param_rand_factor
            )
            f_central_mat = f_central_mat * rand_change[0]
            band_mat = band_mat * rand_change[1]

        fbank_matrix = self._create_fbank_matrix(f_central_mat, band_mat)

        sp_shape = spectrogram.shape

        # Managing multi-channels case (batch, time, channels)
        if len(sp_shape) == 4:
            spectrogram = spectrogram.transpose([0, 3, 1, 2])
            spectrogram = spectrogram.reshape(
                [sp_shape[0] * sp_shape[3], sp_shape[1], sp_shape[2]]
            )

        # FBANK computation
        # print("spectrogram shape: {}".format(spectrogram.shape))
        # print("spectrogram shape: {}".format(fbank_matrix.shape))
        fbanks = paddle.matmul(spectrogram, fbank_matrix)
        if self.log_mel:
            fbanks = self._amplitude_to_DB(fbanks)

        # Reshaping in the case of multi-channel inputs
        if len(sp_shape) == 4:
            fb_shape = fbanks.shape
            fbanks = fbanks.reshape(
                [sp_shape[0], sp_shape[3], fb_shape[1], fb_shape[2]]
            )
            fbanks = fbanks.transpose([0, 2, 3, 1])

        return fbanks

    @staticmethod
    def _to_mel(hz):
        """Returns mel-frequency value corresponding to the input
        frequency value in Hz.

        Arguments
        ---------
        x : float
            The frequency point in Hz.
        """
        return 2595 * math.log10(1 + hz / 700)

    @staticmethod
    def _to_hz(mel):
        """Returns hz-frequency value corresponding to the input
        mel-frequency value.

        Arguments
        ---------
        x : float
            The frequency point in the mel-scale.
        """
        return 700 * (10 ** (mel / 2595) - 1)

    def _triangular_filters(self, all_freqs, f_central, band):
        """Returns fbank matrix using triangular filters.

        Arguments
        ---------
        all_freqs : Tensor
            Tensor gathering all the frequency points.
        f_central : Tensor
            Tensor gathering central frequencies of each filter.
        band : Tensor
            Tensor gathering the bands of each filter.
        """

        # Computing the slops of the filters
        slope = (all_freqs - f_central) / band
        left_side = slope + 1.0
        right_side = -slope + 1.0
        # print("left_side shape: {}".format(left_side.shape))
        # print("right_side shape: {}".format(right_side.shape))
        # Adding zeros for negative values
        zero = paddle.zeros([1])
        min_value = paddle.minimum(left_side, right_side)
        # print("tri shape: {}".format(min_value.shape))
        fbank_matrix = paddle.maximum(
            zero, min_value
        ).transpose(perm=[1, 0])

        return fbank_matrix

    def _rectangular_filters(self, all_freqs, f_central, band):
        """Returns fbank matrix using rectangular filters.

        Arguments
        ---------
        all_freqs : Tensor
            Tensor gathering all the frequency points.
        f_central : Tensor
            Tensor gathering central frequencies of each filter.
        band : Tensor
            Tensor gathering the bands of each filter.
        """

        # cut-off frequencies of the filters
        low_hz = f_central - band
        high_hz = f_central + band

        # Left/right parts of the filter
        left_side = right_size = all_freqs.ge(low_hz)
        right_size = all_freqs.le(high_hz)

        fbank_matrix = (left_side * right_size).float().transpose(0, 1)

        return fbank_matrix

    def _gaussian_filters(
        self, all_freqs, f_central, band, smooth_factor=paddle.to_tensor([2], dtype="float32")
    ):
        """Returns fbank matrix using gaussian filters.

        Arguments
        ---------
        all_freqs : Tensor
            Tensor gathering all the frequency points.
        f_central : Tensor
            Tensor gathering central frequencies of each filter.
        band : Tensor
            Tensor gathering the bands of each filter.
        smooth_factor: Tensor
            Smoothing factor of the gaussian filter. It can be used to employ
            sharper or flatter filters.
        """
        fbank_matrix = paddle.exp(
            -0.5 * ((all_freqs - f_central) / (band / smooth_factor)) ** 2
        ).transpose(0, 1)

        return fbank_matrix

    def _create_fbank_matrix(self, f_central_mat, band_mat):
        """Returns fbank matrix to use for averaging the spectrum with
           the set of filter-banks.

        Arguments
        ---------
        f_central : Tensor
            Tensor gathering central frequencies of each filter.
        band : Tensor
            Tensor gathering the bands of each filter.
        smooth_factor: Tensor
            Smoothing factor of the gaussian filter. It can be used to employ
            sharper or flatter filters.
        """
        if self.filter_shape == "triangular":
            fbank_matrix = self._triangular_filters(
                self.all_freqs_mat, f_central_mat, band_mat
            )

        elif self.filter_shape == "rectangular":
            fbank_matrix = self._rectangular_filters(
                self.all_freqs_mat, f_central_mat, band_mat
            )

        else:
            fbank_matrix = self._gaussian_filters(
                self.all_freqs_mat, f_central_mat, band_mat
            )

        return fbank_matrix

    def _amplitude_to_DB(self, x):
        """Converts  linear-FBANKs to log-FBANKs.

        Arguments
        ---------
        x : Tensor
            A batch of linear FBANK tensors.

        """

        x_db = self.multiplier * paddle.log10(paddle.clip(x, min=self.amin))
        x_db -= self.multiplier * self.db_multiplier

        # Setting up dB max. It is the max over time and frequency,
        # Hence, of a whole sequence (sequence-dependent)
        new_x_db_max = x_db.max(axis=(-2, -1)) - self.top_db

        # Clipping to dB max. The view is necessary as only a scalar is obtained
        # per sequence.
        x_db = paddle.maximum(x_db, new_x_db_max.reshape([x_db.shape[0], 1, 1]))

        return x_db


class DCT(paddle.nn.Layer):
    """Computes the discrete cosine transform.

    This class is primarily used to compute MFCC features of an audio signal
    given a set of FBANK features as input.

    Arguments
    ---------
    input_size : int
        Expected size of the last dimension in the input.
    n_out : int
        Number of output coefficients.
    ortho_norm : bool
        Whether to use orthogonal norm.

    Example
    -------
    >>> import paddle
    >>> inputs = paddle.randn([10, 101, 40])
    >>> compute_mfccs = DCT(input_size=inputs.size(-1))
    >>> features = compute_mfccs(inputs)
    >>> features.shape
    paddle.Size([10, 101, 20])
    """

    def __init__(
        self, input_size, n_out=20, ortho_norm=True,
    ):
        super().__init__()

        if n_out > input_size:
            raise ValueError(
                "Cannot select more DCT coefficients than inputs "
                "(n_out=%i, n_in=%i)" % (n_out, input_size)
            )

        # Generate matix for DCT transformation
        n = paddle.arange(float(input_size))
        k = paddle.arange(float(n_out)).unsqueeze(1)
        dct = paddle.cos(math.pi / float(input_size) * (n + 0.5) * k)

        if ortho_norm:
            dct[0] *= 1.0 / math.sqrt(2.0)
            dct *= math.sqrt(2.0 / float(input_size))
        else:
            dct *= 2.0

        self.dct_mat = dct.t()

    def forward(self, x):
        """Returns the DCT of the input tensor.

        Arguments
        ---------
        x : tensor
            A batch of tensors to transform, usually fbank features.
        """
        # Managing multi-channels case
        input_shape = x.shape
        if len(input_shape) == 4:
            x = x.reshape([x.shape[0] * x.shape[3], x.shape[1], x.shape[2]])

        # apply the DCT transform
        dct = paddle.matmul(x, self.dct_mat)

        # Reshape in the case of multi-channels
        if len(input_shape) == 4:
            dct = dct.reshape(
                [input_shape[0], dct.shape[1], dct.shape[2], input_shape[3]]
            )

        return dct


class Deltas(paddle.nn.Layer):
    """Computes delta coefficients (time derivatives).

    Arguments
    ---------
    win_length : int
        Length of the window used to compute the time derivatives.

    Example
    -------
    >>> inputs = paddle.randn([10, 101, 20])
    >>> compute_deltas = Deltas(input_size=inputs.size(-1))
    >>> features = compute_deltas(inputs)
    >>> features.shape
    paddle.Size([10, 101, 20])
    """

    def __init__(
        self, input_size, window_length=5,
    ):
        super().__init__()
        self.n = (window_length - 1) // 2
        self.denom = self.n * (self.n + 1) * (2 * self.n + 1) / 3
        # deltas = paddle.arange(-self.n, self.n + 1, dtype=paddle.float32,).tile(
        #         [input_size, 1, 1])
        # print("deltas")
        # print(deltas)
        self.register_buffer(
            "kernel",
            paddle.arange(-self.n, self.n + 1, dtype="float32",).tile(
                [input_size, 1, 1]
            ),
        )

    def forward(self, x):
        """Returns the delta coefficients.

        Arguments
        ---------
        x : tensor
            A batch of tensors.
        """
        # Managing multi-channel deltas reshape tensor (batch*channel,time)
        x = x.transpose(perm=[0, 2, 1])
        # print("x shape: {}".format(x.shape))
        or_shape = x.shape
        if len(or_shape) == 4:
            x = x.reshape(or_shape[0] * or_shape[2], or_shape[1], or_shape[3])

        # Padding for time borders
        x = paddle.nn.functional.pad(x, (self.n, self.n), mode="replicate", data_format="NCL")

        # Derivative estimation (with a fixed convolutional kernel)
        delta_coeff = (
            paddle.nn.functional.conv1d(x, self.kernel, groups=x.shape[1])
            / self.denom
        )

        # Retrieving the original dimensionality (for multi-channel case)
        if len(or_shape) == 4:
            delta_coeff = delta_coeff.reshape(
                or_shape[0], or_shape[1], or_shape[2], or_shape[3],
            )
        
        delta_coeff = delta_coeff.transpose(perm=[0, 2, 1])
        return delta_coeff


class ContextWindow(paddle.nn.Layer):
    """Computes the context window.

    This class applies a context window by gathering multiple time steps
    in a single feature vector. The operation is performed with a
    convolutional layer based on a fixed kernel designed for that.

    Arguments
    ---------
    left_frames : int
         Number of left frames (i.e, past frames) to collect.
    right_frames : int
        Number of right frames (i.e, future frames) to collect.

    Example
    -------
    >>> import paddle
    >>> compute_cw = ContextWindow(left_frames=5, right_frames=5)
    >>> inputs = paddle.randn([10, 101, 20])
    >>> features = compute_cw(inputs)
    >>> features.shape
    paddle.Size([10, 101, 220])
    """

    def __init__(
        self, left_frames=0, right_frames=0,
    ):
        super().__init__()
        self.left_frames = left_frames
        self.right_frames = right_frames
        self.context_len = self.left_frames + self.right_frames + 1
        self.kernel_len = 2 * max(self.left_frames, self.right_frames) + 1

        # Kernel definition
        self.kernel = paddle.eye(self.context_len, self.kernel_len)

        if self.right_frames > self.left_frames:
            lag = self.right_frames - self.left_frames
            self.kernel = paddle.roll(self.kernel, lag, 1)

        self.first_call = True

    def forward(self, x):
        """Returns the tensor with the surrounding context.

        Arguments
        ---------
        x : tensor
            A batch of tensors.
        """

        x = x.transpose(perm=[0, 2, 1])

        if self.first_call is True:
            self.first_call = False
            self.kernel = (
                self.kernel.tile([x.shape[1], 1, 1])
                .reshape([x.shape[1] * self.context_len, self.kernel_len],)
                .unsqueeze(1)
            )

        # Managing multi-channel case
        or_shape = x.shape
        if len(or_shape) == 4:
            x = x.reshape(or_shape[0] * or_shape[2], or_shape[1], or_shape[3])

        # Compute context (using the estimated convolutional kernel)
        cw_x = paddle.nn.functional.conv1d(
            x,
            self.kernel,
            groups=x.shape[1],
            padding=max(self.left_frames, self.right_frames),
        )

        # Retrieving the original dimensionality (for multi-channel case)
        if len(or_shape) == 4:
            cw_x = cw_x.reshape(
                or_shape[0], cw_x.shape[1], or_shape[2], cw_x.shape[-1]
            )

        cw_x = cw_x.transpose(perm=[0, 2, 1])

        return cw_x


@register_checkpoint_hooks
class InputNormalization(paddle.nn.Layer):
    """Performs mean and variance normalization of the input tensor.

    Arguments
    ---------
    mean_norm : True
         If True, the mean will be normalized.
    std_norm : True
         If True, the standard deviation will be normalized.
    norm_type : str
         It defines how the statistics are computed ('sentence' computes them
         at sentence level, 'batch' at batch level, 'speaker' at speaker
         level, while global computes a single normalization vector for all
         the sentences in the dataset). Speaker and global statistics are
         computed with a moving average approach.
    avg_factor : float
         It can be used to manually set the weighting factor between
         current statistics and accumulated ones.

    Example
    -------
    >>> import paddle
    >>> norm = InputNormalization()
    >>> inputs = paddle.randn([10, 101, 20])
    >>> inp_len = paddle.ones([10])
    >>> features = norm(inputs, inp_len)
    """

    from typing import Dict

    spk_dict_mean: Dict[int, paddle.Tensor]
    spk_dict_std: Dict[int, paddle.Tensor]
    spk_dict_count: Dict[int, int]

    def __init__(
        self,
        mean_norm=True,
        std_norm=True,
        norm_type="global",
        avg_factor=None,
        requires_grad=False,
        update_until_epoch=3,
    ):
        super().__init__()
        self.mean_norm = mean_norm
        self.std_norm = std_norm
        self.norm_type = norm_type
        self.avg_factor = avg_factor
        self.requires_grad = requires_grad
        self.glob_mean = paddle.to_tensor([0], dtype="float32")
        self.glob_std = paddle.to_tensor([0], dtype="float32")
        self.spk_dict_mean = {}
        self.spk_dict_std = {}
        self.spk_dict_count = {}
        self.weight = 1.0
        self.count = 0
        self.eps = 1e-10
        self.update_until_epoch = update_until_epoch

    def forward(self, x, lengths, spk_ids=paddle.to_tensor([], dtype="float32"), epoch=0):
        """Returns the tensor with the surrounding context.

        Arguments
        ---------
        x : tensor
            A batch of tensors.
        lengths : tensor
            A batch of tensors containing the relative length of each
            sentence (e.g, [0.7, 0.9, 1.0]). It is used to avoid
            computing stats on zero-padded steps.
        spk_ids : tensor containing the ids of each speaker (e.g, [0 10 6]).
            It is used to perform per-speaker normalization when
            norm_type='speaker'.
        """
        N_batches = x.shape[0]

        current_means = []
        current_stds = []

        for snt_id in range(N_batches):

            # Avoiding padded time steps
            actual_size = paddle.round(lengths[snt_id] * x.shape[1]).astype("int32")

            # computing statistics
            current_mean, current_std = self._compute_current_stats(
                x[snt_id, 0:actual_size, ...]
            )

            current_means.append(current_mean)
            current_stds.append(current_std)

            if self.norm_type == "sentence":

                x[snt_id] = (x[snt_id] - current_mean) / current_std

            if self.norm_type == "speaker":

                spk_id = int(spk_ids[snt_id][0])

                if self.training:
                    if spk_id not in self.spk_dict_mean:

                        # Initialization of the dictionary
                        self.spk_dict_mean[spk_id] = current_mean
                        self.spk_dict_std[spk_id] = current_std
                        self.spk_dict_count[spk_id] = 1

                    else:
                        self.spk_dict_count[spk_id] = (
                            self.spk_dict_count[spk_id] + 1
                        )

                        if self.avg_factor is None:
                            self.weight = 1 / self.spk_dict_count[spk_id]
                        else:
                            self.weight = self.avg_factor

                        self.spk_dict_mean[spk_id] = (
                            (1 - self.weight) * self.spk_dict_mean[spk_id]
                            + self.weight * current_mean
                        )
                        self.spk_dict_std[spk_id] = (
                            (1 - self.weight) * self.spk_dict_std[spk_id]
                            + self.weight * current_std
                        )

                        self.spk_dict_mean[spk_id].detach()
                        self.spk_dict_std[spk_id].detach()

                    speaker_mean = self.spk_dict_mean[spk_id].data
                    speaker_std = self.spk_dict_std[spk_id].data
                else:
                    if spk_id in self.spk_dict_mean:
                        speaker_mean = self.spk_dict_mean[spk_id].data
                        speaker_std = self.spk_dict_std[spk_id].data
                    else:
                        speaker_mean = current_mean.data
                        speaker_std = current_std.data

                x[snt_id] = (x[snt_id] - speaker_mean) / speaker_std

        if self.norm_type == "batch" or self.norm_type == "global":
            current_mean = paddle.mean(paddle.stack(current_means), axis=0)
            current_std = paddle.mean(paddle.stack(current_stds), axis=0)

            if self.norm_type == "batch":
                x = (x - current_mean.data) / (current_std.data)

            if self.norm_type == "global":

                if self.training:
                    if self.count == 0:
                        self.glob_mean = current_mean
                        self.glob_std = current_std

                    elif epoch < self.update_until_epoch:
                        if self.avg_factor is None:
                            self.weight = 1 / (self.count + 1)
                        else:
                            self.weight = self.avg_factor

                        self.glob_mean = (
                            1 - self.weight
                        ) * self.glob_mean + self.weight * current_mean

                        self.glob_std = (
                            1 - self.weight
                        ) * self.glob_std + self.weight * current_std

                    self.glob_mean.detach()
                    self.glob_std.detach()

                    self.count = self.count + 1
                # 这里都是去掉了.data
                x = (x - self.glob_mean) / (self.glob_std)

        return x

    def _compute_current_stats(self, x):
        """Returns the tensor with the surrounding context.

        Arguments
        ---------
        x : tensor
            A batch of tensors.
        """
        # Compute current mean
        if self.mean_norm:
            # 这里需要注意一下
            current_mean = paddle.mean(x, axis=0).detach()
        else:
            current_mean = paddle.to_tensor([0.0], dtype="float32")

        # Compute current std
        if self.std_norm:
            current_std = paddle.std(x, axis=0).detach()
        else:
            current_std = paddle.to_tensor([1.0], dtype="float32")

        # Improving numerical stability of std
        current_std = paddle.maximum(
            current_std, self.eps * paddle.ones_like(current_std)
        )

        return current_mean, current_std

    def _statistics_dict(self):
        """Fills the dictionary containing the normalization statistics.
        """
        state = {}
        state["count"] = self.count
        state["glob_mean"] = self.glob_mean
        state["glob_std"] = self.glob_std
        state["spk_dict_mean"] = self.spk_dict_mean
        state["spk_dict_std"] = self.spk_dict_std
        state["spk_dict_count"] = self.spk_dict_count

        return state

    def _load_statistics_dict(self, state):
        """Loads the dictionary containing the statistics.

        Arguments
        ---------
        state : dict
            A dictionary containing the normalization statistics.
        """
        self.count = state["count"]
        if isinstance(state["glob_mean"], int):
            self.glob_mean = state["glob_mean"]
            self.glob_std = state["glob_std"]
        else:
            self.glob_mean = state["glob_mean"]  # .to(self.device_inp)
            self.glob_std = state["glob_std"]  # .to(self.device_inp)

        # Loading the spk_dict_mean in the right device
        self.spk_dict_mean = {}
        for spk in state["spk_dict_mean"]:
            self.spk_dict_mean[spk] = state["spk_dict_mean"][spk]

        # Loading the spk_dict_std in the right device
        self.spk_dict_std = {}
        for spk in state["spk_dict_std"]:
            self.spk_dict_std[spk] = state["spk_dict_std"][spk]

        self.spk_dict_count = state["spk_dict_count"]

        return state

    def to(self, device):
        """Puts the needed tensors in the right device.
        """
        self = super(InputNormalization, self).to(device)
        self.glob_mean = self.glob_mean.to(device)
        self.glob_std = self.glob_std.to(device)
        for spk in self.spk_dict_mean:
            self.spk_dict_mean[spk] = self.spk_dict_mean[spk].to(device)
            self.spk_dict_std[spk] = self.spk_dict_std[spk].to(device)
        return self

    @mark_as_saver
    def _save(self, path):
        """Save statistic dictionary.

        Arguments
        ---------
        path : str
            A path where to save the dictionary.
        """
        stats = self._statistics_dict()
        paddle.save(stats, path)

    @mark_as_transfer
    @mark_as_loader
    def _load(self, path, end_of_epoch=False, device=None):
        """Load statistic dictionary.

        Arguments
        ---------
        path : str
            The path of the statistic dictionary
        device : str, None
            Passed to paddle.load(..., map_location=device)
        """
        del end_of_epoch  # Unused here.
        stats = paddle.load(path, map_location=device)
        self._load_statistics_dict(stats)
