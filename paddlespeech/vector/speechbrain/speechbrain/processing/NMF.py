"""Non-negative matrix factorization

Authors
 * Cem Subakan
"""
import paddle
from speechbrain.processing.features import spectral_magnitude
import speechbrain.processing.features as spf


def spectral_phase(stft, power=2, log=False):
    """Returns the phase of a complex spectrogram.

    Arguments
    ---------
    stft : paddle.Tensor
        A tensor, output from the stft function.

    Example
    -------
    >>> BS, nfft, T = 10, 20, 300
    >>> X_stft = torch.randn(BS, nfft//2 + 1, T, 2)
    >>> phase_mix = spectral_phase(X_stft)
    """

    phase = torch.atan2(stft[:, :, :, 1], stft[:, :, :, 0])

    return phase


def NMF_separate_spectra(Whats, Xmix):
    """This function separates the mixture signals, given NMF template matrices.

    Arguments
    ---------
    Whats : list
        This list contains the list [W1, W2], where W1 W2 are respectively
        the NMF template matrices that correspond to source1 and source2.
        W1, W2 are of size [nfft/2 + 1, K], where nfft is the fft size for STFT,
        and K is the number of vectors (templates) in W.
    Xmix : torch.tensor
        This is the magnitude spectra for the mixtures.
        The size is [BS x T x nfft//2 + 1] where,
        BS = batch size, nfft = fft size, T = number of time steps in the spectra.

    Outputs
    -------
    X1hat : Separated spectrum for source1
        Size = [BS x (nfft/2 +1) x T] where,
        BS = batch size, nfft = fft size, T = number of time steps in the spectra.
    X2hat : Separated Spectrum for source2
        The size definitions are the same as above.

    Example
    --------
    >>> BS, nfft, T = 4, 20, 400
    >>> K1, K2 = 10, 10
    >>> W1hat = torch.randn(nfft//2 + 1, K1)
    >>> W2hat = torch.randn(nfft//2 + 1, K2)
    >>> Whats = [W1hat, W2hat]
    >>> Xmix = torch.randn(BS, T, nfft//2 + 1)
    >>> X1hat, X2hat = NMF_separate_spectra(Whats, Xmix)
    """

    W1, W2 = Whats

    nmixtures = Xmix.shape[0]
    Xmix = Xmix.permute(0, 2, 1).reshape(-1, Xmix.size(-1)).t()
    n = Xmix.shape[1]
    eps = 1e-20

    # Normalize input
    g = Xmix.sum(dim=0) + eps
    z = Xmix / g

    # initialize
    w = torch.cat([W1, W2], dim=1)
    K = w.size(1)
    K1 = W1.size(1)

    h = 0.1 * torch.rand(K, n)
    h /= torch.sum(h, dim=0) + eps

    for ep in range(1000):
        v = z / (torch.matmul(w, h) + eps)

        nh = h * torch.matmul(w.t(), v)
        h = nh / (torch.sum(nh, dim=0) + eps)

    h *= g
    Xhat1 = torch.matmul(w[:, :K1], h[:K1, :])
    Xhat1 = torch.split(Xhat1.unsqueeze(0), Xhat1.size(1) // nmixtures, dim=2)
    Xhat1 = torch.cat(Xhat1, dim=0)

    Xhat2 = torch.matmul(w[:, K1:], h[K1:, :])
    Xhat2 = torch.split(Xhat2.unsqueeze(0), Xhat2.size(1) // nmixtures, dim=2)
    Xhat2 = torch.cat(Xhat2, dim=0)

    return Xhat1, Xhat2


def reconstruct_results(
    X1hat, X2hat, X_stft, sample_rate, win_length, hop_length,
):

    """This function reconstructs the separated spectra into waveforms.

    Arguments
    ---------
    Xhat1 : torch.tensor
        The separated spectrum for source 1 of size [BS, nfft/2 + 1, T],
        where,  BS = batch size, nfft = fft size, T = length of the spectra.
    Xhat2 : torch.tensor
        The separated spectrum for source 2 of size [BS, nfft/2 + 1, T].
        The size definitions are the same as Xhat1.
    X_stft : torch.tensor
        This is the magnitude spectra for the mixtures.
        The size is [BS x nfft//2 + 1 x T x 2] where,
        BS = batch size, nfft = fft size, T = number of time steps in the spectra.
        The last dimension is to represent complex numbers.
    sample_rate : int
        The sampling rate (in Hz) in which we would like to save the results.
    win_length : int
        The length of stft windows (in ms).
    hop_length : int
        The length with which we shift the STFT windows (in ms).

    Returns
    -------
    x1hats : list
        List of waveforms for source 1.
    x2hats : list
        List of waveforms for source 2.

    Example
    -------
    >>> BS, nfft, T = 10, 512, 16000
    >>> sample_rate, win_length, hop_length = 16000, 25, 10
    >>> X1hat = torch.randn(BS, nfft//2 + 1, T)
    >>> X2hat = torch.randn(BS, nfft//2 + 1, T)
    >>> X_stft = torch.randn(BS, nfft//2 + 1, T, 2)
    >>> x1hats, x2hats = reconstruct_results(X1hat, X2hat, X_stft, sample_rate, win_length, hop_length)
    """

    ISTFT = spf.ISTFT(
        sample_rate=sample_rate, win_length=win_length, hop_length=hop_length
    )

    phase_mix = spectral_phase(X_stft)
    mag_mix = spectral_magnitude(X_stft, power=2)

    x1hats, x2hats = [], []
    eps = 1e-25
    for i in range(X1hat.shape[0]):
        X1hat_stft = (
            (X1hat[i] / (eps + X1hat[i] + X2hat[i])).unsqueeze(-1)
            * mag_mix[i].unsqueeze(-1)
            * torch.cat(
                [
                    torch.cos(phase_mix[i].unsqueeze(-1)),
                    torch.sin(phase_mix[i].unsqueeze(-1)),
                ],
                dim=-1,
            )
        )

        X2hat_stft = (
            (X2hat[i] / (eps + X1hat[i] + X2hat[i])).unsqueeze(-1)
            * mag_mix[i].unsqueeze(-1)
            * torch.cat(
                [
                    torch.cos(phase_mix[i].unsqueeze(-1)),
                    torch.sin(phase_mix[i].unsqueeze(-1)),
                ],
                dim=-1,
            )
        )
        X1hat_stft = X1hat_stft.unsqueeze(0).permute(0, 2, 1, 3)
        X2hat_stft = X2hat_stft.unsqueeze(0).permute(0, 2, 1, 3)
        shat1 = ISTFT(X1hat_stft)
        shat2 = ISTFT(X2hat_stft)

        div_factor = 10
        x1 = shat1 / (div_factor * shat1.std())
        x2 = shat2 / (div_factor * shat2.std())

        x1hats.append(x1)
        x2hats.append(x2)
    return x1hats, x2hats
