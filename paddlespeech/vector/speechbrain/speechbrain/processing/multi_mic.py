"""Multi-microphone components.

This library contains functions for multi-microphone signal processing.

Example
-------
>>> import paddle
>>>
>>> from speechbrain.dataio.dataio import read_audio
>>> from speechbrain.processing.features import STFT, ISTFT
>>> from speechbrain.processing.multi_mic import Covariance
>>> from speechbrain.processing.multi_mic import GccPhat, SrpPhat, Music
>>> from speechbrain.processing.multi_mic import DelaySum, Mvdr, Gev
>>>
>>> xs_speech = read_audio(
...    'samples/audio_samples/multi_mic/speech_-0.82918_0.55279_-0.082918.flac'
... )
>>> xs_speech = xs_speech.unsqueeze(0) # [batch, time, channels]
>>> xs_noise_diff = read_audio('samples/audio_samples/multi_mic/noise_diffuse.flac')
>>> xs_noise_diff = xs_noise_diff.unsqueeze(0)
>>> xs_noise_loc = read_audio('samples/audio_samples/multi_mic/noise_0.70225_-0.70225_0.11704.flac')
>>> xs_noise_loc =  xs_noise_loc.unsqueeze(0)
>>> fs = 16000 # sampling rate

>>> ss = xs_speech
>>> nn_diff = 0.05 * xs_noise_diff
>>> nn_loc = 0.05 * xs_noise_loc
>>> xs_diffused_noise = ss + nn_diff
>>> xs_localized_noise = ss + nn_loc

>>> # Delay-and-Sum Beamforming with GCC-PHAT localization
>>> stft = STFT(sample_rate=fs)
>>> cov = Covariance()
>>> gccphat = GccPhat()
>>> delaysum = DelaySum()
>>> istft = ISTFT(sample_rate=fs)

>>> Xs = stft(xs_diffused_noise)
>>> Ns = stft(nn_diff)
>>> XXs = cov(Xs)
>>> NNs = cov(Ns)
>>> tdoas = gccphat(XXs)
>>> Ys_ds = delaysum(Xs, tdoas)
>>> ys_ds = istft(Ys_ds)

>>> # Mvdr Beamforming with SRP-PHAT localization
>>> mvdr = Mvdr()
>>> mics = paddle.zeros((4,3), dtype=paddle.float)
>>> mics[0,:] = paddle.FloatTensor([-0.05, -0.05, +0.00])
>>> mics[1,:] = paddle.FloatTensor([-0.05, +0.05, +0.00])
>>> mics[2,:] = paddle.FloatTensor([+0.05, +0.05, +0.00])
>>> mics[3,:] = paddle.FloatTensor([+0.05, +0.05, +0.00])
>>> srpphat = SrpPhat(mics=mics)
>>> doas = srpphat(XXs)
>>> Ys_mvdr = mvdr(Xs, NNs, doas, doa_mode=True, mics=mics, fs=fs)
>>> ys_mvdr = istft(Ys_mvdr)

>>> # Mvdr Beamforming with MUSIC localization
>>> music = Music(mics=mics)
>>> doas = music(XXs)
>>> Ys_mvdr2 = mvdr(Xs, NNs, doas, doa_mode=True, mics=mics, fs=fs)
>>> ys_mvdr2 = istft(Ys_mvdr2)

>>> # GeV Beamforming
>>> gev = Gev()
>>> Xs = stft(xs_localized_noise)
>>> Ss = stft(ss)
>>> Ns = stft(nn_loc)
>>> SSs = cov(Ss)
>>> NNs = cov(Ns)
>>> Ys_gev = gev(Xs, SSs, NNs)
>>> ys_gev = istft(Ys_gev)

Authors:
 * William Aris
 * Francois Grondin

"""

import paddle
from packaging import version
import speechbrain.processing.decomposition as eig


class Covariance(paddle.nn.Layer):
    """Computes the covariance matrices of the signals.

    Arguments:
    ----------
    average : bool
        Informs the module if it should return an average
        (computed on the time dimension) of the covariance
        matrices. The Default value is True.

    Example
    -------
    >>> import paddle
    >>> from speechbrain.dataio.dataio import read_audio
    >>> from speechbrain.processing.features import STFT
    >>> from speechbrain.processing.multi_mic import Covariance
    >>>
    >>> xs_speech = read_audio(
    ...    'samples/audio_samples/multi_mic/speech_-0.82918_0.55279_-0.082918.flac'
    ... )
    >>> xs_speech = xs_speech.unsqueeze(0) # [batch, time, channels]
    >>> xs_noise = read_audio('samples/audio_samples/multi_mic/noise_diffuse.flac')
    >>> xs_noise = xs_noise.unsqueeze(0)
    >>> xs = xs_speech + 0.05 * xs_noise
    >>> fs = 16000

    >>> stft = STFT(sample_rate=fs)
    >>> cov = Covariance()
    >>>
    >>> Xs = stft(xs)
    >>> XXs = cov(Xs)
    >>> XXs.shape
    paddle.Size([1, 1001, 201, 2, 10])
    """

    def __init__(self, average=True):

        super().__init__()
        self.average = average

    def forward(self, Xs):
        """ This method uses the utility function _cov to compute covariance
        matrices. Therefore, the result has the following format:
        (batch, time_step, n_fft/2 + 1, 2, n_mics + n_pairs).

        The order on the last dimension corresponds to the triu_indices for a
        square matrix. For instance, if we have 4 channels, we get the following
        order: (0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3)
        and (3, 3). Therefore, XXs[..., 0] corresponds to channels (0, 0) and XXs[..., 1]
        corresponds to channels (0, 1).

        Arguments:
        ----------
        Xs : tensor
            A batch of audio signals in the frequency domain.
            The tensor must have the following format:
            (batch, time_step, n_fft/2 + 1, 2, n_mics)
        """

        XXs = Covariance._cov(Xs=Xs, average=self.average)
        return XXs

    @staticmethod
    def _cov(Xs, average=True):
        """ Computes the covariance matrices (XXs) of the signals. The result will
        have the following format: (batch, time_step, n_fft/2 + 1, 2, n_mics + n_pairs).

        Arguments:
        ----------
        Xs : tensor
            A batch of audio signals in the frequency domain.
            The tensor must have the following format:
            (batch, time_step, n_fft/2 + 1, 2, n_mics)

        average : boolean
            Informs the function if it should return an average
            (computed on the time dimension) of the covariance
            matrices. Default value is True.
        """

        # Get useful dimensions
        n_mics = Xs.shape[4]

        # Formatting the real and imaginary parts
        Xs_re = Xs[..., 0, :].unsqueeze(4)
        Xs_im = Xs[..., 1, :].unsqueeze(4)

        # Computing the covariance
        Rxx_re = paddle.matmul(Xs_re, Xs_re.transpose(3, 4)) + paddle.matmul(
            Xs_im, Xs_im.transpose(3, 4)
        )

        Rxx_im = paddle.matmul(Xs_re, Xs_im.transpose(3, 4)) - paddle.matmul(
            Xs_im, Xs_re.transpose(3, 4)
        )

        # Selecting the upper triangular part of the covariance matrices
        idx = paddle.triu_indices(n_mics, n_mics)

        XXs_re = Rxx_re[..., idx[0], idx[1]]
        XXs_im = Rxx_im[..., idx[0], idx[1]]

        XXs = paddle.stack((XXs_re, XXs_im), 3)

        # Computing the average if desired
        if average is True:
            n_time_frames = XXs.shape[1]
            XXs = paddle.mean(XXs, 1, keepdim=True)
            XXs = XXs.repeat(1, n_time_frames, 1, 1, 1)

        return XXs


class DelaySum(paddle.nn.Layer):
    """Performs delay and sum beamforming by using the TDOAs and
        the first channel as a reference.

        Example
        -------
        >>> import paddle

        >>> from speechbrain.dataio.dataio import read_audio
        >>> from speechbrain.processing.features import STFT, ISTFT
        >>> from speechbrain.processing.multi_mic import Covariance
        >>> from speechbrain.processing.multi_mic import GccPhat, DelaySum
        >>>
        >>> xs_speech = read_audio(
        ...    'samples/audio_samples/multi_mic/speech_-0.82918_0.55279_-0.082918.flac'
        ... )
        >>> xs_speech = xs_speech. unsqueeze(0) # [batch, time, channel]
        >>> xs_noise  = read_audio('samples/audio_samples/multi_mic/noise_diffuse.flac')
        >>> xs_noise = xs_noise.unsqueeze(0) #[batch, time, channels]
        >>> fs = 16000
        >>> xs = xs_speech + 0.05 * xs_noise
        >>>
        >>> stft = STFT(sample_rate=fs)
        >>> cov = Covariance()
        >>> gccphat = GccPhat()
        >>> delaysum = DelaySum()
        >>> istft = ISTFT(sample_rate=fs)
        >>>
        >>> Xs = stft(xs)
        >>> XXs = cov(Xs)
        >>> tdoas = gccphat(XXs)
        >>> Ys = delaysum(Xs, tdoas)
        >>> ys = istft(Ys)
    """

    def __init__(self):

        super().__init__()

    def forward(
        self,
        Xs,
        localization_tensor,
        doa_mode=False,
        mics=None,
        fs=None,
        c=343.0,
    ):
        """This method computes a steering vector by using the TDOAs/DOAs and
        then calls the utility function _delaysum to perform beamforming.
        The result has the following format: (batch, time_step, n_fft, 2, 1).

        Arguments
        ---------
        Xs : tensor
            A batch of audio signals in the frequency domain.
            The tensor must have the following format:
            (batch, time_step, n_fft/2 + 1, 2, n_mics)
        localization_tensor : tensor
            A tensor containing either time differences of arrival (TDOAs)
            (in samples) for each timestamp or directions of arrival (DOAs)
            (xyz coordinates in meters). If localization_tensor represents
            TDOAs, then its format is (batch, time_steps, n_mics + n_pairs).
            If localization_tensor represents DOAs, then its format is
            (batch, time_steps, 3)
        doa_mode : bool
            The user needs to set this parameter to True if localization_tensor
            represents DOAs instead of TDOAs. Its default value is set to False.
        mics : tensor
            The cartesian position (xyz coordinates in meters) of each microphone.
            The tensor must have the following format (n_mics, 3). This
            parameter is only mandatory when localization_tensor represents
            DOAs.
        fs : int
            The sample rate in Hertz of the signals. This parameter is only
            mandatory when localization_tensor represents DOAs.
        c : float
            The speed of sound in the medium. The speed is expressed in meters
            per second and the default value of this parameter is 343 m/s. This
            parameter is only used when localization_tensor represents DOAs.
        """

        # Get useful dimensions
        n_fft = Xs.shape[2]
        localization_tensor = localization_tensor.to(Xs.device)
        # Convert the tdoas to taus
        if doa_mode:
            taus = doas2taus(doas=localization_tensor, mics=mics, fs=fs, c=c)

        else:
            taus = tdoas2taus(tdoas=localization_tensor)

        # Generate the steering vector
        As = steering(taus=taus, n_fft=n_fft)

        # Apply delay and sum
        Ys = DelaySum._delaysum(Xs=Xs, As=As)

        return Ys

    @staticmethod
    def _delaysum(Xs, As):
        """Perform delay and sum beamforming. The result has
        the following format: (batch, time_step, n_fft, 2, 1).

        Arguments
        ---------
        Xs : tensor
            A batch of audio signals in the frequency domain.
            The tensor must have the following format:
            (batch, time_step, n_fft/2 + 1, 2, n_mics)
        As : tensor
            The steering vector to point in the direction of
            the target source. The tensor must have the format
            (batch, time_step, n_fft/2 + 1, 2, n_mics)
        """

        # Get useful dimensions
        n_mics = Xs.shape[4]

        # Generate unmixing coefficients
        Ws_re = As[..., 0, :] / n_mics
        Ws_im = -1 * As[..., 1, :] / n_mics

        # Get input signal
        Xs_re = Xs[..., 0, :]
        Xs_im = Xs[..., 1, :]

        # Applying delay and sum
        Ys_re = paddle.sum((Ws_re * Xs_re - Ws_im * Xs_im), dim=3, keepdim=True)
        Ys_im = paddle.sum((Ws_re * Xs_im + Ws_im * Xs_re), dim=3, keepdim=True)

        # Assembling the result
        Ys = paddle.stack((Ys_re, Ys_im), 3)

        return Ys


class Mvdr(paddle.nn.Layer):
    """Perform minimum variance distortionless response (MVDR) beamforming
    by using an input signal in the frequency domain, its covariance matrices
    and tdoas (to compute a steering vector).

        Example
        -------
        >>> import paddle

        >>> from speechbrain.dataio.dataio import read_audio
        >>> from speechbrain.processing.features import STFT, ISTFT
        >>> from speechbrain.processing.multi_mic import Covariance
        >>> from speechbrain.processing.multi_mic import GccPhat, DelaySum
        >>>
        >>> xs_speech = read_audio(
        ...    'samples/audio_samples/multi_mic/speech_-0.82918_0.55279_-0.082918.flac'
        ... )
        >>> xs_speech = xs_speech.unsqueeze(0) # [batch, time, channel]
        >>> xs_noise  = read_audio('samples/audio_samples/multi_mic/noise_diffuse.flac')
        >>> xs_noise = xs_noise.unsqueeze(0) #[batch, time, channels]
        >>> fs = 16000
        >>> xs = xs_speech + 0.05 * xs_noise
        >>>
        >>> stft = STFT(sample_rate=fs)
        >>> cov = Covariance()
        >>> gccphat = GccPhat()
        >>> mvdr = Mvdr()
        >>> istft = ISTFT(sample_rate=fs)
        >>>
        >>> Xs = stft(xs)
        >>> Ns = stft(xs_noise)
        >>> XXs = cov(Xs)
        >>> NNs = cov(Ns)
        >>> tdoas = gccphat(XXs)
        >>> Ys = mvdr(Xs, NNs, tdoas)
        >>> ys = istft(Ys)
    """

    def __init__(self, eps=1e-20):

        super().__init__()

        self.eps = eps

    def forward(
        self,
        Xs,
        NNs,
        localization_tensor,
        doa_mode=False,
        mics=None,
        fs=None,
        c=343.0,
    ):
        """This method computes a steering vector before using the
        utility function _mvdr to perform beamforming. The result has
        the following format: (batch, time_step, n_fft, 2, 1).

        Arguments
        ---------
        Xs : tensor
            A batch of audio signals in the frequency domain.
            The tensor must have the following format:
            (batch, time_step, n_fft/2 + 1, 2, n_mics)
        NNs : tensor
            The covariance matrices of the noise signal. The tensor must
            have the format (batch, time_steps, n_fft/2 + 1, 2, n_mics + n_pairs)
        localization_tensor : tensor
            A tensor containing either time differences of arrival (TDOAs)
            (in samples) for each timestamp or directions of arrival (DOAs)
            (xyz coordinates in meters). If localization_tensor represents
            TDOAs, then its format is (batch, time_steps, n_mics + n_pairs).
            If localization_tensor represents DOAs, then its format is
            (batch, time_steps, 3)
        doa_mode : bool
            The user needs to set this parameter to True if localization_tensor
            represents DOAs instead of TDOAs. Its default value is set to False.
        mics : tensor
            The cartesian position (xyz coordinates in meters) of each microphone.
            The tensor must have the following format (n_mics, 3). This
            parameter is only mandatory when localization_tensor represents
            DOAs.
        fs : int
            The sample rate in Hertz of the signals. This parameter is only
            mandatory when localization_tensor represents DOAs.
        c : float
            The speed of sound in the medium. The speed is expressed in meters
            per second and the default value of this parameter is 343 m/s. This
            parameter is only used when localization_tensor represents DOAs.
        """
        # Get useful dimensions
        n_fft = Xs.shape[2]
        localization_tensor = localization_tensor.to(Xs.device)
        NNs = NNs.to(Xs.device)
        if mics is not None:
            mics = mics.to(Xs.device)

        # Convert the tdoas to taus
        if doa_mode:
            taus = doas2taus(doas=localization_tensor, mics=mics, fs=fs, c=c)

        else:
            taus = tdoas2taus(tdoas=localization_tensor)

        # Generate the steering vector
        As = steering(taus=taus, n_fft=n_fft)

        # Perform mvdr
        Ys = Mvdr._mvdr(Xs=Xs, NNs=NNs, As=As)

        return Ys

    @staticmethod
    def _mvdr(Xs, NNs, As, eps=1e-20):
        """Perform minimum variance distortionless response beamforming.

        Arguments
        ---------
        Xs : tensor
            A batch of audio signals in the frequency domain.
            The tensor must have the following format:
            (batch, time_step, n_fft/2 + 1, 2, n_mics).
        NNs : tensor
            The covariance matrices of the noise signal. The tensor must
            have the format (batch, time_steps, n_fft/2 + 1, 2, n_mics + n_pairs).
        As : tensor
            The steering vector to point in the direction of
            the target source. The tensor must have the format
            (batch, time_step, n_fft/2 + 1, 2, n_mics).
        """

        # Get unique covariance values to reduce the number of computations
        NNs_val, NNs_idx = paddle.unique(NNs, return_inverse=True, dim=1)

        # Inverse covariance matrices
        NNs_inv = eig.inv(NNs_val)

        # Capture real and imaginary parts, and restore time steps
        NNs_inv_re = NNs_inv[..., 0][:, NNs_idx]
        NNs_inv_im = NNs_inv[..., 1][:, NNs_idx]

        # Decompose steering vector
        AsC_re = As[..., 0, :].unsqueeze(4)
        AsC_im = 1.0 * As[..., 1, :].unsqueeze(4)
        AsT_re = AsC_re.transpose(3, 4)
        AsT_im = -1.0 * AsC_im.transpose(3, 4)

        # Project
        NNs_inv_AsC_re = paddle.matmul(NNs_inv_re, AsC_re) - paddle.matmul(
            NNs_inv_im, AsC_im
        )
        NNs_inv_AsC_im = paddle.matmul(NNs_inv_re, AsC_im) + paddle.matmul(
            NNs_inv_im, AsC_re
        )

        # Compute the gain
        alpha = 1.0 / (
            paddle.matmul(AsT_re, NNs_inv_AsC_re)
            - paddle.matmul(AsT_im, NNs_inv_AsC_im)
        )

        # Get the unmixing coefficients
        Ws_re = paddle.matmul(NNs_inv_AsC_re, alpha).squeeze(4)
        Ws_im = -paddle.matmul(NNs_inv_AsC_im, alpha).squeeze(4)

        # Applying MVDR
        Xs_re = Xs[..., 0, :]
        Xs_im = Xs[..., 1, :]

        Ys_re = paddle.sum((Ws_re * Xs_re - Ws_im * Xs_im), dim=3, keepdim=True)
        Ys_im = paddle.sum((Ws_re * Xs_im + Ws_im * Xs_re), dim=3, keepdim=True)

        Ys = paddle.stack((Ys_re, Ys_im), -2)

        return Ys


class Gev(paddle.nn.Layer):
    """Generalized EigenValue decomposition (GEV) Beamforming.

    Example
    -------
    >>> from speechbrain.dataio.dataio import read_audio
    >>> import paddle
    >>>
    >>> from speechbrain.processing.features import STFT, ISTFT
    >>> from speechbrain.processing.multi_mic import Covariance
    >>> from speechbrain.processing.multi_mic import Gev
    >>>
    >>> xs_speech = read_audio(
    ...    'samples/audio_samples/multi_mic/speech_-0.82918_0.55279_-0.082918.flac'
    ... )
    >>> xs_speech  = xs_speech.unsqueeze(0) # [batch, time, channels]
    >>> xs_noise = read_audio('samples/audio_samples/multi_mic/noise_0.70225_-0.70225_0.11704.flac')
    >>> xs_noise = xs_noise.unsqueeze(0)
    >>> fs = 16000
    >>> ss = xs_speech
    >>> nn = 0.05 * xs_noise
    >>> xs = ss + nn
    >>>
    >>> stft = STFT(sample_rate=fs)
    >>> cov = Covariance()
    >>> gev = Gev()
    >>> istft = ISTFT(sample_rate=fs)
    >>>
    >>> Ss = stft(ss)
    >>> Nn = stft(nn)
    >>> Xs = stft(xs)
    >>>
    >>> SSs = cov(Ss)
    >>> NNs = cov(Nn)
    >>>
    >>> Ys = gev(Xs, SSs, NNs)
    >>> ys = istft(Ys)
    """

    def __init__(self):

        super().__init__()

    def forward(self, Xs, SSs, NNs):
        """ This method uses the utility function _gev to perform generalized
        eigenvalue decomposition beamforming. Therefore, the result has
        the following format: (batch, time_step, n_fft, 2, 1).

        Arguments
        ---------
        Xs : tensor
            A batch of audio signals in the frequency domain.
            The tensor must have the following format:
            (batch, time_step, n_fft/2 + 1, 2, n_mics).
        SSs : tensor
            The covariance matrices of the target signal. The tensor must
            have the format (batch, time_steps, n_fft/2 + 1, 2, n_mics + n_pairs).
        NNs : tensor
            The covariance matrices of the noise signal. The tensor must
            have the format (batch, time_steps, n_fft/2 + 1, 2, n_mics + n_pairs).
        """

        Ys = Gev._gev(Xs=Xs, SSs=SSs, NNs=NNs)

        return Ys

    @staticmethod
    def _gev(Xs, SSs, NNs):
        """ Perform generalized eigenvalue decomposition beamforming. The result
        has the following format: (batch, time_step, n_fft, 2, 1).

        Arguments
        ---------
        Xs : tensor
            A batch of audio signals in the frequency domain.
            The tensor must have the following format:
            (batch, time_step, n_fft/2 + 1, 2, n_mics).
        SSs : tensor
            The covariance matrices of the target signal. The tensor must
            have the format (batch, time_steps, n_fft/2 + 1, 2, n_mics + n_pairs).
        NNs : tensor
            The covariance matrices of the noise signal. The tensor must
            have the format (batch, time_steps, n_fft/2 + 1, 2, n_mics + n_pairs).
        """

        # Putting on the right device
        SSs = SSs.to(Xs.device)
        NNs = NNs.to(Xs.device)

        # Get useful dimensions
        n_mics = Xs.shape[4]
        n_mics_pairs = SSs.shape[4]

        # Computing the eigenvectors
        SSs_NNs = paddle.cat((SSs, NNs), dim=4)
        SSs_NNs_val, SSs_NNs_idx = paddle.unique(
            SSs_NNs, return_inverse=True, dim=1
        )

        SSs = SSs_NNs_val[..., range(0, n_mics_pairs)]
        NNs = SSs_NNs_val[..., range(n_mics_pairs, 2 * n_mics_pairs)]
        NNs = eig.pos_def(NNs)
        Vs, Ds = eig.gevd(SSs, NNs)

        # Beamforming
        F_re = Vs[..., (n_mics - 1), 0]
        F_im = Vs[..., (n_mics - 1), 1]

        # Normalize
        F_norm = 1.0 / (
            paddle.sum(F_re ** 2 + F_im ** 2, dim=3, keepdim=True) ** 0.5
        ).repeat(1, 1, 1, n_mics)
        F_re *= F_norm
        F_im *= F_norm

        Ws_re = F_re[:, SSs_NNs_idx]
        Ws_im = F_im[:, SSs_NNs_idx]

        Xs_re = Xs[..., 0, :]
        Xs_im = Xs[..., 1, :]

        Ys_re = paddle.sum((Ws_re * Xs_re - Ws_im * Xs_im), dim=3, keepdim=True)
        Ys_im = paddle.sum((Ws_re * Xs_im + Ws_im * Xs_re), dim=3, keepdim=True)

        # Assembling the output
        Ys = paddle.stack((Ys_re, Ys_im), 3)

        return Ys


class GccPhat(paddle.nn.Layer):
    """Generalized Cross-Correlation with Phase Transform localization.

    Arguments
    ---------
    tdoa_max : int
        Specifies a range to search for delays. For example, if
        tdoa_max = 10, the method will restrict its search for delays
        between -10 and 10 samples. This parameter is optional and its
        default value is None. When tdoa_max is None, the method will
        search for delays between -n_fft/2 and n_fft/2 (full range).
    eps : float
        A small value to avoid divisions by 0 with the phase transformation.
        The default value is 1e-20.

    Example
    -------
    >>> import paddle

    >>> from speechbrain.dataio.dataio import read_audio
    >>> from speechbrain.processing.features import STFT, ISTFT
    >>> from speechbrain.processing.multi_mic import Covariance
    >>> from speechbrain.processing.multi_mic import GccPhat, DelaySum
    >>>
    >>> xs_speech = read_audio(
    ...    'samples/audio_samples/multi_mic/speech_-0.82918_0.55279_-0.082918.flac'
    ... )
    >>> xs_speech = xs_speech.unsqueeze(0) # [batch, time, channel]
    >>> xs_noise  = read_audio('samples/audio_samples/multi_mic/noise_diffuse.flac')
    >>> xs_noise = xs_noise.unsqueeze(0) #[batch, time, channels]
    >>> fs = 16000
    >>> xs = xs_speech + 0.05 * xs_noise
    >>>
    >>> stft = STFT(sample_rate=fs)
    >>> cov = Covariance()
    >>> gccphat = GccPhat()
    >>> Xs = stft(xs)
    >>> XXs = cov(Xs)
    >>> tdoas = gccphat(XXs)
    """

    def __init__(self, tdoa_max=None, eps=1e-20):

        super().__init__()
        self.tdoa_max = tdoa_max
        self.eps = eps

    def forward(self, XXs):
        """ Perform generalized cross-correlation with phase transform localization
        by using the utility function _gcc_phat and by extracting the delays (in samples)
        before performing a quadratic interpolation to improve the accuracy.
        The result has the format: (batch, time_steps, n_mics + n_pairs).

        The order on the last dimension corresponds to the triu_indices for a
        square matrix. For instance, if we have 4 channels, we get the following
        order: (0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3)
        and (3, 3). Therefore, delays[..., 0] corresponds to channels (0, 0) and delays[..., 1]
        corresponds to channels (0, 1).

        Arguments:
        ----------
        XXs : tensor
            The covariance matrices of the input signal. The tensor must
            have the format (batch, time_steps, n_fft/2 + 1, 2, n_mics + n_pairs).
        """

        xxs = GccPhat._gcc_phat(XXs=XXs, eps=self.eps)
        delays = GccPhat._extract_delays(xxs=xxs, tdoa_max=self.tdoa_max)
        tdoas = GccPhat._interpolate(xxs=xxs, delays=delays)
        return tdoas

    @staticmethod
    def _gcc_phat(XXs, eps=1e-20):
        """ Evaluate GCC-PHAT for each timestamp. It returns the result in the time
        domain. The result has the format: (batch, time_steps, n_fft, n_mics + n_pairs).

        Arguments
        ---------
        XXs : tensor
            The covariance matrices of the input signal. The tensor must
            have the format (batch, time_steps, n_fft/2 + 1, 2, n_mics + n_pairs).
        eps : float
            A small value to avoid divisions by 0 with the phase transform. The
            default value is 1e-20.
        """

        # Get useful dimensions
        n_samples = (XXs.shape[2] - 1) * 2

        # Extracting the tensors needed
        XXs_val, XXs_idx = paddle.unique(XXs, return_inverse=True, dim=4)

        XXs_re = XXs_val[..., 0, :]
        XXs_im = XXs_val[..., 1, :]

        # Applying the phase transform
        XXs_abs = paddle.sqrt(XXs_re ** 2 + XXs_im ** 2) + eps
        XXs_re_phat = XXs_re / XXs_abs
        XXs_im_phat = XXs_im / XXs_abs
        XXs_phat = paddle.stack((XXs_re_phat, XXs_im_phat), 4)

        # Returning in the temporal domain
        XXs_phat = XXs_phat.transpose(2, 3)

        if version.parse(paddle.__version__) >= version.parse("1.8.0"):
            XXs_phat = paddle.complex(XXs_phat[..., 0], XXs_phat[..., 1])
            xxs = paddle.fft.irfft(XXs_phat, n=n_samples)
        else:
            xxs = paddle.irfft(XXs_phat, signal_ndim=1, signal_sizes=[n_samples])

        xxs = xxs[..., XXs_idx, :]

        # Formatting the output
        xxs = xxs.transpose(2, 3)

        return xxs

    @staticmethod
    def _extract_delays(xxs, tdoa_max=None):
        """ Extract the rounded delays from the cross-correlation for each timestamp.
        The result has the format: (batch, time_steps, n_mics + n_pairs).

        Arguments
        ---------
        xxs : tensor
            The correlation signals obtained after a gcc-phat operation. The tensor
            must have the format (batch, time_steps, n_fft, n_mics + n_pairs).
        tdoa_max : int
            Specifies a range to search for delays. For example, if
            tdoa_max = 10, the method will restrict its search for delays
            between -10 and 10 samples. This parameter is optional and its
            default value is None. When tdoa_max is None, the method will
            search for delays between -n_fft/2 and +n_fft/2 (full range).
        """

        # Get useful dimensions
        n_fft = xxs.shape[2]

        # If no tdoa specified, cover the whole frame
        if tdoa_max is None:
            tdoa_max = paddle.div(n_fft, 2, rounding_mode="floor")

        # Splitting the GCC-PHAT values to search in the range
        slice_1 = xxs[..., 0:tdoa_max, :]
        slice_2 = xxs[..., -tdoa_max:, :]

        xxs_sliced = paddle.cat((slice_1, slice_2), 2)

        # Extracting the delays in the range
        _, delays = paddle.max(xxs_sliced, 2)

        # Adjusting the delays that were affected by the slicing
        offset = n_fft - xxs_sliced.shape[2]
        idx = delays >= slice_1.shape[2]
        delays[idx] += offset

        # Centering the delays around 0
        delays[idx] -= n_fft

        return delays

    @staticmethod
    def _interpolate(xxs, delays):
        """Perform quadratic interpolation on the cross-correlation to
        improve the tdoa accuracy. The result has the format:
        (batch, time_steps, n_mics + n_pairs)

        Arguments
        ---------
        xxs : tensor
            The correlation signals obtained after a gcc-phat operation. The tensor
            must have the format (batch, time_steps, n_fft, n_mics + n_pairs).
        delays : tensor
            The rounded tdoas obtained by selecting the sample with the highest
            amplitude. The tensor must have the format
            (batch, time_steps, n_mics + n_pairs).
        """

        # Get useful dimensions
        n_fft = xxs.shape[2]

        # Get the max amplitude and its neighbours
        tp = paddle.fmod((delays - 1) + n_fft, n_fft).unsqueeze(2)
        y1 = paddle.gather(xxs, 2, tp).squeeze(2)
        tp = paddle.fmod(delays + n_fft, n_fft).unsqueeze(2)
        y2 = paddle.gather(xxs, 2, tp).squeeze(2)
        tp = paddle.fmod((delays + 1) + n_fft, n_fft).unsqueeze(2)
        y3 = paddle.gather(xxs, 2, tp).squeeze(2)

        # Add a fractional part to the initially rounded delay
        delays_frac = delays + (y1 - y3) / (2 * y1 - 4 * y2 + 2 * y3)

        return delays_frac


class SrpPhat(paddle.nn.Layer):
    """Steered-Response Power with Phase Transform Localization.

    Arguments
    ---------
    mics : tensor
        The cartesian coordinates (xyz) in meters of each microphone.
        The tensor must have the following format (n_mics, 3).
    space : string
        If this parameter is set to 'sphere', the localization will
        be done in 3D by searching in a sphere of possible doas. If
        it set to 'circle', the search will be done in 2D by searching
        in a circle. By default, this parameter is set to 'sphere'.
        Note: The 'circle' option isn't implemented yet.
    sample_rate : int
        The sample rate in Hertz of the signals to perform SRP-PHAT on.
        By default, this parameter is set to 16000 Hz.
    speed_sound : float
        The speed of sound in the medium. The speed is expressed in meters
        per second and the default value of this parameter is 343 m/s.
    eps : float
        A small value to avoid errors like division by 0. The default value
        of this parameter is 1e-20.

    Example
    -------
    >>> import paddle

    >>> from speechbrain.dataio.dataio import read_audio
    >>> from speechbrain.processing.features import STFT
    >>> from speechbrain.processing.multi_mic import Covariance
    >>> from speechbrain.processing.multi_mic import SrpPhat

    >>> xs_speech = read_audio('samples/audio_samples/multi_mic/speech_-0.82918_0.55279_-0.082918.flac')
    >>> xs_noise = read_audio('samples/audio_samples/multi_mic/noise_diffuse.flac')
    >>> fs = 16000

    >>> xs_speech = xs_speech.unsqueeze(0) # [batch, time, channels]
    >>> xs_noise = xs_noise.unsqueeze(0)

    >>> ss1 = xs_speech
    >>> ns1 = 0.05 * xs_noise
    >>> xs1 = ss1 + ns1

    >>> ss2 = xs_speech
    >>> ns2 = 0.20 * xs_noise
    >>> xs2 = ss2 + ns2

    >>> ss = paddle.cat((ss1,ss2), dim=0)
    >>> ns = paddle.cat((ns1,ns2), dim=0)
    >>> xs = paddle.cat((xs1,xs2), dim=0)

    >>> mics = paddle.zeros((4,3), dtype=paddle.float)
    >>> mics[0,:] = paddle.FloatTensor([-0.05, -0.05, +0.00])
    >>> mics[1,:] = paddle.FloatTensor([-0.05, +0.05, +0.00])
    >>> mics[2,:] = paddle.FloatTensor([+0.05, +0.05, +0.00])
    >>> mics[3,:] = paddle.FloatTensor([+0.05, +0.05, +0.00])

    >>> stft = STFT(sample_rate=fs)
    >>> cov = Covariance()
    >>> srpphat = SrpPhat(mics=mics)

    >>> Xs = stft(xs)
    >>> XXs = cov(Xs)
    >>> doas = srpphat(XXs)
    """

    def __init__(
        self,
        mics,
        space="sphere",
        sample_rate=16000,
        speed_sound=343.0,
        eps=1e-20,
    ):

        super().__init__()

        # Generate the doas
        if space == "sphere":
            self.doas = sphere()

        if space == "circle":
            pass

        # Generate associated taus with the doas
        self.taus = doas2taus(
            self.doas, mics=mics, fs=sample_rate, c=speed_sound
        )

        # Save epsilon
        self.eps = eps

    def forward(self, XXs):
        """ Perform SRP-PHAT localization on a signal by computing a steering
        vector and then by using the utility function _srp_phat to extract the doas.
        The result is a tensor containing the directions of arrival (xyz coordinates
        (in meters) in the direction of the sound source). The output tensor
        has the format (batch, time_steps, 3).

        This localization method uses Global Coherence Field (GCF):
        https://www.researchgate.net/publication/221491705_Speaker_localization_based_on_oriented_global_coherence_field

        Arguments
        ---------
        XXs : tensor
            The covariance matrices of the input signal. The tensor must
            have the format (batch, time_steps, n_fft/2 + 1, 2, n_mics + n_pairs).
        """
        # Get useful dimensions
        n_fft = XXs.shape[2]

        # Generate the steering vector
        As = steering(self.taus.to(XXs.device), n_fft)

        # Perform srp-phat
        doas = SrpPhat._srp_phat(XXs=XXs, As=As, doas=self.doas, eps=self.eps)

        return doas

    @staticmethod
    def _srp_phat(XXs, As, doas, eps=1e-20):
        """Perform srp-phat to find the direction of arrival
        of the sound source. The result is a tensor containing the directions
        of arrival (xyz coordinates (in meters) in the direction of the sound source).
        The output tensor has the format: (batch, time_steps, 3).

        Arguments
        ---------
        XXs : tensor
            The covariance matrices of the input signal. The tensor must
            have the format (batch, time_steps, n_fft/2 + 1, 2, n_mics + n_pairs).
        As : tensor
            The steering vector that cover the all the potential directions
            of arrival. The tensor must have the format
            (n_doas, n_fft/2 + 1, 2, n_mics).
        doas : tensor
            All the possible directions of arrival that will be scanned. The
            tensor must have the format (n_doas, 3).
        """

        # Putting on the right device
        As = As.to(XXs.device)
        doas = doas.to(XXs.device)

        # Get useful dimensions
        n_mics = As.shape[3]

        # Get the indices for the pairs of microphones
        idx = paddle.triu_indices(n_mics, n_mics)

        # Generate the demixing vector from the steering vector
        As_1_re = As[:, :, 0, idx[0, :]]
        As_1_im = As[:, :, 1, idx[0, :]]
        As_2_re = As[:, :, 0, idx[1, :]]
        As_2_im = As[:, :, 1, idx[1, :]]
        Ws_re = As_1_re * As_2_re + As_1_im * As_2_im
        Ws_im = As_1_re * As_2_im - As_1_im * As_2_re
        Ws_re = Ws_re.reshape(Ws_re.shape[0], -1)
        Ws_im = Ws_im.reshape(Ws_im.shape[0], -1)

        # Get unique covariance values to reduce the number of computations
        XXs_val, XXs_idx = paddle.unique(XXs, return_inverse=True, dim=1)

        # Perform the phase transform
        XXs_re = XXs_val[:, :, :, 0, :]
        XXs_im = XXs_val[:, :, :, 1, :]
        XXs_re = XXs_re.reshape((XXs_re.shape[0], XXs_re.shape[1], -1))
        XXs_im = XXs_im.reshape((XXs_im.shape[0], XXs_im.shape[1], -1))
        XXs_abs = paddle.sqrt(XXs_re ** 2 + XXs_im ** 2) + eps
        XXs_re_norm = XXs_re / XXs_abs
        XXs_im_norm = XXs_im / XXs_abs

        # Project on the demixing vectors, and keep only real part
        Ys_A = paddle.matmul(XXs_re_norm, Ws_re.transpose(0, 1))
        Ys_B = paddle.matmul(XXs_im_norm, Ws_im.transpose(0, 1))
        Ys = Ys_A - Ys_B

        # Get maximum points
        _, doas_idx = paddle.max(Ys, dim=2)

        # Repeat for each frame
        doas = (doas[doas_idx, :])[:, XXs_idx, :]

        return doas


class Music(paddle.nn.Layer):
    """Multiple Signal Classification (MUSIC) localization.

    Arguments
    ---------
    mics : tensor
        The cartesian coordinates (xyz) in meters of each microphone.
        The tensor must have the following format (n_mics, 3).
    space : string
        If this parameter is set to 'sphere', the localization will
        be done in 3D by searching in a sphere of possible doas. If
        it set to 'circle', the search will be done in 2D by searching
        in a circle. By default, this parameter is set to 'sphere'.
        Note: The 'circle' option isn't implemented yet.
    sample_rate : int
        The sample rate in Hertz of the signals to perform SRP-PHAT on.
        By default, this parameter is set to 16000 Hz.
    speed_sound : float
        The speed of sound in the medium. The speed is expressed in meters
        per second and the default value of this parameter is 343 m/s.
    eps : float
        A small value to avoid errors like division by 0. The default value
        of this parameter is 1e-20.
    n_sig : int
        An estimation of the number of sound sources. The default value is set
        to one source.

    Example
    -------
    >>> import paddle

    >>> from speechbrain.dataio.dataio import read_audio
    >>> from speechbrain.processing.features import STFT
    >>> from speechbrain.processing.multi_mic import Covariance
    >>> from speechbrain.processing.multi_mic import SrpPhat

    >>> xs_speech = read_audio('samples/audio_samples/multi_mic/speech_-0.82918_0.55279_-0.082918.flac')
    >>> xs_noise = read_audio('samples/audio_samples/multi_mic/noise_diffuse.flac')
    >>> fs = 16000

    >>> xs_speech = xs_speech.unsqueeze(0) # [batch, time, channels]
    >>> xs_noise = xs_noise.unsqueeze(0)

    >>> ss1 = xs_speech
    >>> ns1 = 0.05 * xs_noise
    >>> xs1 = ss1 + ns1

    >>> ss2 = xs_speech
    >>> ns2 = 0.20 * xs_noise
    >>> xs2 = ss2 + ns2

    >>> ss = paddle.cat((ss1,ss2), dim=0)
    >>> ns = paddle.cat((ns1,ns2), dim=0)
    >>> xs = paddle.cat((xs1,xs2), dim=0)

    >>> mics = paddle.zeros((4,3), dtype=paddle.float)
    >>> mics[0,:] = paddle.FloatTensor([-0.05, -0.05, +0.00])
    >>> mics[1,:] = paddle.FloatTensor([-0.05, +0.05, +0.00])
    >>> mics[2,:] = paddle.FloatTensor([+0.05, +0.05, +0.00])
    >>> mics[3,:] = paddle.FloatTensor([+0.05, +0.05, +0.00])

    >>> stft = STFT(sample_rate=fs)
    >>> cov = Covariance()
    >>> music = Music(mics=mics)

    >>> Xs = stft(xs)
    >>> XXs = cov(Xs)
    >>> doas = music(XXs)
    """

    def __init__(
        self,
        mics,
        space="sphere",
        sample_rate=16000,
        speed_sound=343.0,
        eps=1e-20,
        n_sig=1,
    ):

        super().__init__()

        # Generate the doas
        if space == "sphere":
            self.doas = sphere()

        if space == "circle":
            pass

        # Generate associated taus with the doas
        self.taus = doas2taus(
            self.doas, mics=mics, fs=sample_rate, c=speed_sound
        )

        # Save epsilon
        self.eps = eps

        # Save number of signals
        self.n_sig = n_sig

    def forward(self, XXs):
        """Perform MUSIC localization on a signal by computing a steering
        vector and then by using the utility function _music to extract the doas.
        The result is a tensor containing the directions of arrival (xyz coordinates
        (in meters) in the direction of the sound source). The output tensor
        has the format (batch, time_steps, 3).

        Arguments
        ---------
        XXs : tensor
            The covariance matrices of the input signal. The tensor must
            have the format (batch, time_steps, n_fft/2 + 1, 2, n_mics + n_pairs).
        """

        # Get useful dimensions
        n_fft = XXs.shape[2]

        # Generate the steering vector
        As = steering(self.taus.to(XXs.device), n_fft)

        # Perform music
        doas = Music._music(
            XXs=XXs, As=As, doas=self.doas, n_sig=self.n_sig, eps=self.eps
        )

        return doas

    @staticmethod
    def _music(XXs, As, doas, n_sig, eps=1e-20):
        """Perform multiple signal classification to find the
        direction of arrival of the sound source. The result
        has the format: (batch, time_steps, 3).

        Arguments
        ---------
        XXs : tensor
            The covariance matrices of the input signal. The tensor must
            have the format (batch, time_steps, n_fft/2 + 1, 2, n_mics + n_pairs).
        As : tensor
            The steering vector that covers the all the potential directions
            of arrival. The tensor must have the format.
            (n_doas, n_fft/2 + 1, 2, n_mics).
        doas : tensor
            All the possible directions of arrival that will be scanned. The
            tensor must have the format (n_doas, 3).
        n_sig : int
            The number of signals in the signal + noise subspace (default is 1).
        """

        # Putting on the right device
        As = As.to(XXs.device)
        doas = doas.to(XXs.device)

        # Collecting data
        n_mics = As.shape[3]
        n_doas = As.shape[0]
        n_bins = As.shape[2]
        svd_range = n_mics - n_sig

        # Get unique values to reduce computations
        XXs_val, XXs_idx = paddle.unique(XXs, return_inverse=True, dim=1)

        # Singular value decomposition
        Us, _ = eig.svdl(XXs_val)

        # Format for the projection
        Us = Us.unsqueeze(2).repeat(1, 1, n_doas, 1, 1, 1, 1)
        Us_re = Us[..., range(0, svd_range), 0]
        Us_im = Us[..., range(0, svd_range), 1]

        # Fixing the format of the steering vector
        As = (
            As.unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(6)
            .permute(0, 1, 2, 3, 6, 5, 4)
        )
        As = As.repeat(Us.shape[0], Us.shape[1], 1, 1, 1, 1, 1)

        As_re = As[..., 0]
        As_im = As[..., 1]

        # Applying MUSIC's formula
        As_mm_Us_re = paddle.matmul(As_re, Us_re) + paddle.matmul(As_im, Us_im)
        As_mm_Us_im = paddle.matmul(As_re, Us_im) - paddle.matmul(As_im, Us_re)

        As_mm_Us_abs = paddle.sqrt(As_mm_Us_re ** 2 + As_mm_Us_im ** 2)
        As_mm_Us_sum = paddle.sum(As_mm_Us_abs, dim=5)

        As_As_abs = paddle.sum(As_re ** 2, dim=5) + paddle.sum(As_im ** 2, dim=5)

        Ps = (As_As_abs / (As_mm_Us_sum + eps)).squeeze(4)

        Ys = paddle.sum(Ps, dim=3) / n_bins

        # Get maximum points
        _, doas_idx = paddle.max(Ys, dim=2)

        doas = (doas[doas_idx, :])[:, XXs_idx, :]

        return doas


def doas2taus(doas, mics, fs, c=343.0):
    """This function converts directions of arrival (xyz coordinates
    expressed in meters) in time differences of arrival (expressed in
    samples). The result has the following format: (batch, time_steps, n_mics).

    Arguments
    ---------
    doas : tensor
        The directions of arrival expressed with cartesian coordinates (xyz)
        in meters. The tensor must have the following format: (batch, time_steps, 3).
    mics : tensor
        The cartesian position (xyz) in meters of each microphone.
        The tensor must have the following format (n_mics, 3).
    fs : int
        The sample rate in Hertz of the signals.
    c : float
        The speed of sound in the medium. The speed is expressed in meters
        per second and the default value of this parameter is 343 m/s.

    Example
    -------
    >>> import paddle

    >>> from speechbrain.dataio.dataio import read_audio
    >>> from speechbrain.processing.multi_mic import sphere, doas2taus

    >>> xs = read_audio('samples/audio_samples/multi_mic/speech_-0.82918_0.55279_-0.082918.flac')
    >>> xs = xs.unsqueeze(0) # [batch, time, channels]
    >>> fs = 16000
    >>> mics = paddle.zeros((4,3), dtype=paddle.float)
    >>> mics[0,:] = paddle.FloatTensor([-0.05, -0.05, +0.00])
    >>> mics[1,:] = paddle.FloatTensor([-0.05, +0.05, +0.00])
    >>> mics[2,:] = paddle.FloatTensor([+0.05, +0.05, +0.00])
    >>> mics[3,:] = paddle.FloatTensor([+0.05, +0.05, +0.00])

    >>> doas = sphere()
    >>> taus = doas2taus(doas, mics, fs)
    """

    taus = (fs / c) * paddle.matmul(doas.to(mics.device), mics.transpose(0, 1))

    return taus


def tdoas2taus(tdoas):
    """ This function selects the tdoas of each channel and put them
    in a tensor. The result has the following format:
    (batch, time_steps, n_mics).

    Arguments:
    ----------
    tdoas : tensor
       The time difference of arrival (TDOA) (in samples) for
       each timestamp. The tensor has the format
       (batch, time_steps, n_mics + n_pairs).

    Example
    -------
    >>> import paddle
    >>> from speechbrain.dataio.dataio import read_audio
    >>> from speechbrain.processing.features import STFT
    >>> from speechbrain.processing.multi_mic import Covariance
    >>> from speechbrain.processing.multi_mic import GccPhat, tdoas2taus
    >>>
    >>> xs_speech = read_audio(
    ...    'samples/audio_samples/multi_mic/speech_-0.82918_0.55279_-0.082918.flac'
    ... )
    >>> xs_noise = read_audio('samples/audio_samples/multi_mic/noise_diffuse.flac')
    >>> xs = xs_speech + 0.05 * xs_noise
    >>> xs = xs.unsqueeze(0)
    >>> fs = 16000
    >>>
    >>> stft = STFT(sample_rate=fs)
    >>> cov = Covariance()
    >>> gccphat = GccPhat()
    >>>
    >>> Xs = stft(xs)
    >>> XXs = cov(Xs)
    >>> tdoas = gccphat(XXs)
    >>> taus = tdoas2taus(tdoas)
    """

    n_pairs = tdoas.shape[len(tdoas.shape) - 1]
    n_channels = int(((1 + 8 * n_pairs) ** 0.5 - 1) / 2)
    taus = tdoas[..., range(0, n_channels)]

    return taus


def steering(taus, n_fft):
    """ This function computes a steering vector by using the time differences
    of arrival for each channel (in samples) and the number of bins (n_fft).
    The result has the following format: (batch, time_step, n_fft/2 + 1, 2, n_mics).

    Arguments:
    ----------
    taus : tensor
        The time differences of arrival for each channel. The tensor must have
        the following format: (batch, time_steps, n_mics).

    n_fft : int
        The number of bins resulting of the STFT. It is assumed that the
        argument "onesided" was set to True for the STFT.

    Example:
    --------f
    >>> import paddle
    >>> from speechbrain.dataio.dataio import read_audio
    >>> from speechbrain.processing.features import STFT
    >>> from speechbrain.processing.multi_mic import Covariance
    >>> from speechbrain.processing.multi_mic import GccPhat, tdoas2taus, steering
    >>>
    >>> xs_speech = read_audio(
    ...    'samples/audio_samples/multi_mic/speech_-0.82918_0.55279_-0.082918.flac'
    ... )
    >>> xs_noise = read_audio('samples/audio_samples/multi_mic/noise_diffuse.flac')
    >>> xs = xs_speech + 0.05 * xs_noise
    >>> xs = xs.unsqueeze(0) # [batch, time, channels]
    >>> fs = 16000

    >>> stft = STFT(sample_rate=fs)
    >>> cov = Covariance()
    >>> gccphat = GccPhat()
    >>>
    >>> Xs = stft(xs)
    >>> n_fft = Xs.shape[2]
    >>> XXs = cov(Xs)
    >>> tdoas = gccphat(XXs)
    >>> taus = tdoas2taus(tdoas)
    >>> As = steering(taus, n_fft)
    """

    # Collecting useful numbers
    pi = 3.141592653589793

    frame_size = int((n_fft - 1) * 2)

    # Computing the different parts of the steering vector
    omegas = 2 * pi * paddle.arange(0, n_fft, device=taus.device) / frame_size
    omegas = omegas.repeat(taus.shape + (1,))
    taus = taus.unsqueeze(len(taus.shape)).repeat(
        (1,) * len(taus.shape) + (n_fft,)
    )

    # Assembling the steering vector
    a_re = paddle.cos(-omegas * taus)
    a_im = paddle.sin(-omegas * taus)
    a = paddle.stack((a_re, a_im), len(a_re.shape))
    a = a.transpose(len(a.shape) - 3, len(a.shape) - 1).transpose(
        len(a.shape) - 3, len(a.shape) - 2
    )

    return a


def sphere(levels_count=4):
    """ This function generates cartesian coordinates (xyz) for a set
    of points forming a 3D sphere. The coordinates are expressed in
    meters and can be used as doas. The result has the format:
    (n_points, 3).

    Arguments
    ---------
    levels_count : int
        A number proportional to the number of points that the user
        wants to generate.
            - If levels_count = 1, then the sphere will have 42 points
            - If levels_count = 2, then the sphere will have 162 points
            - If levels_count = 3, then the sphere will have 642 points
            - If levels_count = 4, then the sphere will have 2562 points
            - If levels_count = 5, then the sphere will have 10242 points
            - ...
        By default, levels_count is set to 4.

    Example
    -------
    >>> import paddle
    >>> from speechbrain.processing.multi_mic import sphere
    >>> doas = sphere()
    """

    # Generate points at level 0

    h = (5.0 ** 0.5) / 5.0
    r = (2.0 / 5.0) * (5.0 ** 0.5)
    pi = 3.141592654

    pts = paddle.zeros((12, 3), dtype=paddle.float)
    pts[0, :] = paddle.FloatTensor([0, 0, 1])
    pts[11, :] = paddle.FloatTensor([0, 0, -1])
    pts[range(1, 6), 0] = r * paddle.sin(2.0 * pi * paddle.arange(0, 5) / 5.0)
    pts[range(1, 6), 1] = r * paddle.cos(2.0 * pi * paddle.arange(0, 5) / 5.0)
    pts[range(1, 6), 2] = h
    pts[range(6, 11), 0] = (
        -1.0 * r * paddle.sin(2.0 * pi * paddle.arange(0, 5) / 5.0)
    )
    pts[range(6, 11), 1] = (
        -1.0 * r * paddle.cos(2.0 * pi * paddle.arange(0, 5) / 5.0)
    )
    pts[range(6, 11), 2] = -1.0 * h

    # Generate triangles at level 0

    trs = paddle.zeros((20, 3), dtype=paddle.long)

    trs[0, :] = paddle.LongTensor([0, 2, 1])
    trs[1, :] = paddle.LongTensor([0, 3, 2])
    trs[2, :] = paddle.LongTensor([0, 4, 3])
    trs[3, :] = paddle.LongTensor([0, 5, 4])
    trs[4, :] = paddle.LongTensor([0, 1, 5])

    trs[5, :] = paddle.LongTensor([9, 1, 2])
    trs[6, :] = paddle.LongTensor([10, 2, 3])
    trs[7, :] = paddle.LongTensor([6, 3, 4])
    trs[8, :] = paddle.LongTensor([7, 4, 5])
    trs[9, :] = paddle.LongTensor([8, 5, 1])

    trs[10, :] = paddle.LongTensor([4, 7, 6])
    trs[11, :] = paddle.LongTensor([5, 8, 7])
    trs[12, :] = paddle.LongTensor([1, 9, 8])
    trs[13, :] = paddle.LongTensor([2, 10, 9])
    trs[14, :] = paddle.LongTensor([3, 6, 10])

    trs[15, :] = paddle.LongTensor([11, 6, 7])
    trs[16, :] = paddle.LongTensor([11, 7, 8])
    trs[17, :] = paddle.LongTensor([11, 8, 9])
    trs[18, :] = paddle.LongTensor([11, 9, 10])
    trs[19, :] = paddle.LongTensor([11, 10, 6])

    # Generate next levels

    for levels_index in range(0, levels_count):

        #      0
        #     / \
        #    A---B
        #   / \ / \
        #  1---C---2

        trs_count = trs.shape[0]
        subtrs_count = trs_count * 4

        subtrs = paddle.zeros((subtrs_count, 6), dtype=paddle.long)

        subtrs[0 * trs_count + paddle.arange(0, trs_count), 0] = trs[:, 0]
        subtrs[0 * trs_count + paddle.arange(0, trs_count), 1] = trs[:, 0]
        subtrs[0 * trs_count + paddle.arange(0, trs_count), 2] = trs[:, 0]
        subtrs[0 * trs_count + paddle.arange(0, trs_count), 3] = trs[:, 1]
        subtrs[0 * trs_count + paddle.arange(0, trs_count), 4] = trs[:, 2]
        subtrs[0 * trs_count + paddle.arange(0, trs_count), 5] = trs[:, 0]

        subtrs[1 * trs_count + paddle.arange(0, trs_count), 0] = trs[:, 0]
        subtrs[1 * trs_count + paddle.arange(0, trs_count), 1] = trs[:, 1]
        subtrs[1 * trs_count + paddle.arange(0, trs_count), 2] = trs[:, 1]
        subtrs[1 * trs_count + paddle.arange(0, trs_count), 3] = trs[:, 1]
        subtrs[1 * trs_count + paddle.arange(0, trs_count), 4] = trs[:, 1]
        subtrs[1 * trs_count + paddle.arange(0, trs_count), 5] = trs[:, 2]

        subtrs[2 * trs_count + paddle.arange(0, trs_count), 0] = trs[:, 2]
        subtrs[2 * trs_count + paddle.arange(0, trs_count), 1] = trs[:, 0]
        subtrs[2 * trs_count + paddle.arange(0, trs_count), 2] = trs[:, 1]
        subtrs[2 * trs_count + paddle.arange(0, trs_count), 3] = trs[:, 2]
        subtrs[2 * trs_count + paddle.arange(0, trs_count), 4] = trs[:, 2]
        subtrs[2 * trs_count + paddle.arange(0, trs_count), 5] = trs[:, 2]

        subtrs[3 * trs_count + paddle.arange(0, trs_count), 0] = trs[:, 0]
        subtrs[3 * trs_count + paddle.arange(0, trs_count), 1] = trs[:, 1]
        subtrs[3 * trs_count + paddle.arange(0, trs_count), 2] = trs[:, 1]
        subtrs[3 * trs_count + paddle.arange(0, trs_count), 3] = trs[:, 2]
        subtrs[3 * trs_count + paddle.arange(0, trs_count), 4] = trs[:, 2]
        subtrs[3 * trs_count + paddle.arange(0, trs_count), 5] = trs[:, 0]

        subtrs_flatten = paddle.cat(
            (subtrs[:, [0, 1]], subtrs[:, [2, 3]], subtrs[:, [4, 5]]), axis=0
        )
        subtrs_sorted, _ = paddle.sort(subtrs_flatten, axis=1)

        index_max = paddle.max(subtrs_sorted)

        subtrs_scalar = (
            subtrs_sorted[:, 0] * (index_max + 1) + subtrs_sorted[:, 1]
        )

        unique_scalar, unique_indices = paddle.unique(
            subtrs_scalar, return_inverse=True
        )

        unique_values = paddle.zeros(
            (unique_scalar.shape[0], 2), dtype=unique_scalar.dtype
        )

        unique_values[:, 0] = paddle.div(
            unique_scalar, index_max + 1, rounding_mode="floor"
        )
        unique_values[:, 1] = unique_scalar - unique_values[:, 0] * (
            index_max + 1
        )

        trs = paddle.transpose(paddle.reshape(unique_indices, (3, -1)), 0, 1)

        pts = pts[unique_values[:, 0], :] + pts[unique_values[:, 1], :]
        pts /= paddle.repeat_interleave(
            paddle.unsqueeze(paddle.sum(pts ** 2, axis=1) ** 0.5, 1), 3, 1
        )

    return pts
