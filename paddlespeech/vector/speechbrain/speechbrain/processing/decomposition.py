"""
Generalized Eigenvalue Decomposition.

This library contains different methods to adjust the format of
complex Hermitian matrices and find their eigenvectors and
eigenvalues.

Authors
 * William Aris 2020
 * Francois Grondin 2020
"""

import paddle


def gevd(a, b=None):
    """This method computes the eigenvectors and the eigenvalues
    of complex Hermitian matrices. The method finds a solution to
    the problem AV = BVD where V are the eigenvectors and D are
    the eigenvalues.

    The eigenvectors returned by the method (vs) are stored in a tensor
    with the following format (*,C,C,2).

    The eigenvalues returned by the method (ds) are stored in a tensor
    with the following format (*,C,C,2).

    Arguments
    ---------
    a : tensor
        A first input matrix. It is equivalent to the matrix A in the
        equation in the description above. The tensor must have the
        following format: (*,2,C+P).

    b : tensor
        A second input matrix. It is equivalent tot the matrix B in the
        equation in the description above. The tensor must have the
        following format: (*,2,C+P).
        This argument is optional and its default value is None. If
        b == None, then b is replaced by the identity matrix in the
        computations.

    Example
    -------

    Suppose we would like to compute eigenvalues/eigenvectors on the
    following complex Hermitian matrix:

    A = [ 52        34 + 37j  16 + j28 ;
          34 - 37j  125       41 + j3  ;
          16 - 28j  41 - j3   62       ]

    >>> a = torch.FloatTensor([[52,34,16,125,41,62],[0,37,28,0,3,0]])
    >>> vs, ds = gevd(a)

    This corresponds to:

    D = [ 20.9513  0        0        ;
          0        43.9420  0        ;
          0        0        174.1067 ]

    V = [ 0.085976 - 0.85184j  -0.24620 + 0.12244j  -0.24868 - 0.35991j  ;
          -0.16006 + 0.20244j   0.37084 + 0.40173j  -0.79175 - 0.087312j ;
          -0.43990 + 0.082884j  -0.36724 - 0.70045j -0.41728 + 0 j       ]

    where

    A = VDV^-1

    """

    # Dimensions
    D = a.dim()
    P = a.shape[D - 1]
    C = int(round(((1 + 8 * P) ** 0.5 - 1) / 2))

    # Converting the input matrices to block matrices
    ash = f(a)

    if b is None:

        b = torch.zeros(a.shape, dtype=a.dtype, device=a.device)
        ids = torch.triu_indices(C, C)
        b[..., 0, ids[0] == ids[1]] = 1.0

    bsh = f(b)

    # Performing the Cholesky decomposition
    lsh = torch.linalg.cholesky(bsh)
    lsh_inv = torch.inverse(lsh)
    lsh_inv_T = torch.transpose(lsh_inv, D - 2, D - 1)

    # Computing the matrix C
    csh = torch.matmul(lsh_inv, torch.matmul(ash, lsh_inv_T))

    # Performing the eigenvalue decomposition
    es, ysh = torch.linalg.eigh(csh, UPLO="U")

    # Collecting the eigenvalues
    dsh = torch.zeros(
        a.shape[slice(0, D - 2)] + (2 * C, 2 * C),
        dtype=a.dtype,
        device=a.device,
    )
    dsh[..., range(0, 2 * C), range(0, 2 * C)] = es

    # Collecting the eigenvectors
    vsh = torch.matmul(lsh_inv_T, ysh)

    # Converting the block matrices to full complex matrices
    vs = ginv(vsh)
    ds = ginv(dsh)

    return vs, ds


def svdl(a):
    """ Singular Value Decomposition (Left Singular Vectors).

    This function finds the eigenvalues and eigenvectors of the
    input multiplied by its transpose (a x a.T).

    The function will return (in this order):
        1. The eigenvalues in a tensor with the format (*,C,C,2)
        2. The eigenvectors in a tensor with the format (*,C,C,2)

    Arguments:
    ----------
    a : tensor
        A complex input matrix to work with. The tensor must have
        the following format: (*,2,C+P).

    Example:
    --------
    >>> import paddle

    >>> from speechbrain.processing.features import STFT
    >>> from speechbrain.processing.multi_mic import Covariance
    >>> from speechbrain.processing.decomposition import svdl
    >>> from speechbrain.dataio.dataio import read_audio_multichannel

    >>> xs_speech = read_audio_multichannel(
    ...    'samples/audio_samples/multi_mic/speech_-0.82918_0.55279_-0.082918.flac'
    ... )
    >>> xs_noise = read_audio_multichannel('samples/audio_samples/multi_mic/noise_diffuse.flac')
    >>> xs = xs_speech + 0.05 * xs_noise
    >>> xs = xs.unsqueeze(0).float()
    >>>
    >>> stft = STFT(sample_rate=16000)
    >>> cov = Covariance()
    >>>
    >>> Xs = stft(xs)
    >>> XXs = cov(Xs)
    >>> us, ds = svdl(XXs)
    """

    # Dimensions
    D = a.dim()
    P = a.shape[D - 1]
    C = int(round(((1 + 8 * P) ** 0.5 - 1) / 2))

    # Computing As * As_T
    ash = f(a)
    ash_T = torch.transpose(ash, -2, -1)

    ash_mm_ash_T = torch.matmul(ash, ash_T)

    # Finding the eigenvectors and eigenvalues
    es, ush = torch.linalg.eigh(ash_mm_ash_T, UPLO="U")

    # Collecting the eigenvalues
    dsh = torch.zeros(ush.shape, dtype=es.dtype, device=es.device)
    dsh[..., range(0, 2 * C), range(0, 2 * C)] = torch.sqrt(es)

    # Converting the block matrices to full complex matrices
    us = ginv(ush)
    ds = ginv(dsh)

    return us, ds


def f(ws):
    """Transform 1.

    This method takes a complex Hermitian matrix represented by its
    upper triangular part and converts it to a block matrix
    representing the full original matrix with real numbers.
    The output tensor will have the following format:
    (*,2C,2C)

    Arguments
    ---------
    ws : tensor
        An input matrix. The tensor must have the following format:
        (*,2,C+P)
    """

    # Dimensions
    D = ws.dim()
    ws = ws.transpose(D - 2, D - 1)
    P = ws.shape[D - 2]
    C = int(round(((1 + 8 * P) ** 0.5 - 1) / 2))

    # Output matrix
    wsh = torch.zeros(
        ws.shape[0 : (D - 2)] + (2 * C, 2 * C),
        dtype=ws.dtype,
        device=ws.device,
    )
    ids = torch.triu_indices(C, C)
    wsh[..., ids[1] * 2, ids[0] * 2] = ws[..., 0]
    wsh[..., ids[0] * 2, ids[1] * 2] = ws[..., 0]
    wsh[..., ids[1] * 2 + 1, ids[0] * 2 + 1] = ws[..., 0]
    wsh[..., ids[0] * 2 + 1, ids[1] * 2 + 1] = ws[..., 0]
    wsh[..., ids[0] * 2, ids[1] * 2 + 1] = -1 * ws[..., 1]
    wsh[..., ids[1] * 2 + 1, ids[0] * 2] = -1 * ws[..., 1]
    wsh[..., ids[0] * 2 + 1, ids[1] * 2] = ws[..., 1]
    wsh[..., ids[1] * 2, ids[0] * 2 + 1] = ws[..., 1]

    return wsh


def finv(wsh):
    """ Inverse transform 1

    This method takes a block matrix representing a complex Hermitian
    matrix and converts it to a complex matrix represented by its
    upper triangular part. The result will have the following format:
    (*,2,C+P)

    Arguments
    ---------
    wsh : tensor
        An input matrix. The tensor must have the following format:
        (*,2C,2C)
    """

    # Dimensions
    D = wsh.dim()
    C = int(wsh.shape[D - 1] / 2)
    P = int(C * (C + 1) / 2)

    # Output matrix
    ws = torch.zeros(
        wsh.shape[0 : (D - 2)] + (2, P), dtype=wsh.dtype, device=wsh.device
    )
    ids = torch.triu_indices(C, C)
    ws[..., 0, :] = wsh[..., ids[0] * 2, ids[1] * 2]
    ws[..., 1, :] = -1 * wsh[..., ids[0] * 2, ids[1] * 2 + 1]

    return ws


def g(ws):
    """Transform 2.

    This method takes a full complex matrix and converts it to a block
    matrix. The result will have the following format:
    (*,2C,2C).

    Arguments
    ---------
    ws : tensor
        An input matrix. The tensor must have the following format:
        (*,C,C,2)
    """

    # Dimensions
    D = ws.dim()
    C = ws.shape[D - 2]

    # Output matrix
    wsh = torch.zeros(
        ws.shape[0 : (D - 3)] + (2 * C, 2 * C),
        dtype=ws.dtype,
        device=ws.device,
    )
    wsh[..., slice(0, 2 * C, 2), slice(0, 2 * C, 2)] = ws[..., 0]
    wsh[..., slice(1, 2 * C, 2), slice(1, 2 * C, 2)] = ws[..., 0]
    wsh[..., slice(0, 2 * C, 2), slice(1, 2 * C, 2)] = -1 * ws[..., 1]
    wsh[..., slice(1, 2 * C, 2), slice(0, 2 * C, 2)] = ws[..., 1]

    return wsh


def ginv(wsh):
    """Inverse transform 2.

    This method takes a complex Hermitian matrix represented by a block
    matrix and converts it to a full complex complex matrix. The
    result will have the following format:
    (*,C,C,2)

    Arguments
    ---------
    wsh : tensor
        An input matrix. The tensor must have the following format:
        (*,2C,2C)
    """

    # Extracting data
    D = wsh.dim()
    C = int(wsh.shape[D - 1] / 2)

    # Output matrix
    ws = torch.zeros(
        wsh.shape[0 : (D - 2)] + (C, C, 2), dtype=wsh.dtype, device=wsh.device
    )
    ws[..., 0] = wsh[..., slice(0, 2 * C, 2), slice(0, 2 * C, 2)]
    ws[..., 1] = wsh[..., slice(1, 2 * C, 2), slice(0, 2 * C, 2)]

    return ws


def pos_def(ws, alpha=0.001, eps=1e-20):
    """Diagonal modification.

    This method takes a complex Hermitian matrix represented by its upper
    triangular part and adds the value of its trace multiplied by alpha
    to the real part of its diagonal. The output will have the format:
    (*,2,C+P)

    Arguments
    ---------
    ws : tensor
        An input matrix. The tensor must have the following format:
        (*,2,C+P)

    alpha : float
        A coefficient to multiply the trace. The default value is 0.001.

    eps : float
        A small value to increase the real part of the diagonal. The
        default value is 1e-20.
    """

    # Extracting data
    D = ws.dim()
    P = ws.shape[D - 1]
    C = int(round(((1 + 8 * P) ** 0.5 - 1) / 2))

    # Finding the indices of the diagonal
    ids_triu = torch.triu_indices(C, C)
    ids_diag = torch.eq(ids_triu[0, :], ids_triu[1, :])

    # Computing the trace
    trace = torch.sum(ws[..., 0, ids_diag], D - 2)
    trace = trace.view(trace.shape + (1,))
    trace = trace.repeat((1,) * (D - 2) + (C,))

    # Adding the trace multiplied by alpha to the diagonal
    ws_pf = ws.clone()
    ws_pf[..., 0, ids_diag] += alpha * trace + eps

    return ws_pf


def inv(x):
    """Inverse Hermitian Matrix.

    This method finds the inverse of a complex Hermitian matrix
    represented by its upper triangular part. The result will have
    the following format: (*, C, C, 2).

    Arguments
    ---------
    x : tensor
        An input matrix to work with. The tensor must have the
        following format: (*, 2, C+P)

    Example
    -------
    >>> import paddle
    >>>
    >>> from speechbrain.dataio.dataio import read_audio
    >>> from speechbrain.processing.features import STFT
    >>> from speechbrain.processing.multi_mic import Covariance
    >>> from speechbrain.processing.decomposition import inv
    >>>
    >>> xs_speech = read_audio(
    ...    'samples/audio_samples/multi_mic/speech_-0.82918_0.55279_-0.082918.flac'
    ... )
    >>> xs_noise = read_audio('samples/audio_samples/multi_mic/noise_0.70225_-0.70225_0.11704.flac')
    >>> xs = xs_speech + 0.05 * xs_noise
    >>> xs = xs.unsqueeze(0).float()
    >>>
    >>> stft = STFT(sample_rate=16000)
    >>> cov = Covariance()
    >>>
    >>> Xs = stft(xs)
    >>> XXs = cov(Xs)
    >>> XXs_inv = inv(XXs)
    """

    # Dimensions
    d = x.dim()
    p = x.shape[-1]
    n_channels = int(round(((1 + 8 * p) ** 0.5 - 1) / 2))

    # Output matrix
    ash = f(pos_def(x))
    ash_inv = torch.inverse(ash)
    as_inv = finv(ash_inv)

    indices = torch.triu_indices(n_channels, n_channels)

    x_inv = torch.zeros(
        x.shape[slice(0, d - 2)] + (n_channels, n_channels, 2),
        dtype=x.dtype,
        device=x.device,
    )

    x_inv[..., indices[1], indices[0], 0] = as_inv[..., 0, :]
    x_inv[..., indices[1], indices[0], 1] = -1 * as_inv[..., 1, :]
    x_inv[..., indices[0], indices[1], 0] = as_inv[..., 0, :]
    x_inv[..., indices[0], indices[1], 1] = as_inv[..., 1, :]

    return x_inv
