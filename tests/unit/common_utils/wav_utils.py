from typing import Optional

import scipy.io.wavfile
import paddle

def normalize_wav(tensor: paddle.Tensor) -> paddle.Tensor:
    if tensor.dtype == paddle.float32:
        pass
    elif tensor.dtype == paddle.int32:
        tensor = paddle.cast(tensor, paddle.float32)
        tensor[tensor > 0] /= 2147483647.0
        tensor[tensor < 0] /= 2147483648.0
    elif tensor.dtype == paddle.int16:
        tensor = paddle.cast(tensor, paddle.float32)
        tensor[tensor > 0] /= 32767.0
        tensor[tensor < 0] /= 32768.0
    elif tensor.dtype == paddle.uint8:
        tensor = paddle.cast(tensor, paddle.float32) - 128
        tensor[tensor > 0] /= 127.0
        tensor[tensor < 0] /= 128.0
    return tensor


def get_wav_data(
    dtype: str,
    num_channels: int,
    *,
    num_frames: Optional[int] = None,
    normalize: bool = True,
    channels_first: bool = True,
):
    """Generate linear signal of the given dtype and num_channels

    Data range is
        [-1.0, 1.0] for float32,
        [-2147483648, 2147483647] for int32
        [-32768, 32767] for int16
        [0, 255] for uint8

    num_frames allow to change the linear interpolation parameter.
    Default values are 256 for uint8, else 1 << 16.
    1 << 16 as default is so that int16 value range is completely covered.
    """
    dtype_ = getattr(paddle, dtype)

    if num_frames is None:
        if dtype == "uint8":
            num_frames = 256
        else:
            num_frames = 1 << 16

    # paddle linspace not support uint8, int8, int16
    #if dtype == "uint8":
    #    base = paddle.linspace(0, 255, num_frames, dtype=dtype_)
    #elif dtype == "int8":
    #    base = paddle.linspace(-128, 127, num_frames, dtype=dtype_)
    if dtype == "float32":
        base = paddle.linspace(-1.0, 1.0, num_frames, dtype=dtype_)
    elif dtype == "float64":
        base = paddle.linspace(-1.0, 1.0, num_frames, dtype=dtype_)
    elif dtype == "int32":
        base = paddle.linspace(-2147483648, 2147483647, num_frames, dtype=dtype_)
    #elif dtype == "int16":
    #    base = paddle.linspace(-32768, 32767, num_frames, dtype=dtype_)
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")
    data = base.tile([num_channels, 1])
    if not channels_first:
        data = data.transpose([1, 0])
    if normalize:
        data = normalize_wav(data)
    return data


def load_wav(path: str, normalize=True, channels_first=True) -> paddle.Tensor:
    """Load wav file without paddleaudio"""
    sample_rate, data = scipy.io.wavfile.read(path)
    data = paddle.to_tensor(data.copy())
    if data.ndim == 1:
        data = data.unsqueeze(1)
    if normalize:
        data = normalize_wav(data)
    if channels_first:
        data = data.transpose([1, 0])
    return data, sample_rate


def save_wav(path, data, sample_rate, channels_first=True):
    """Save wav file without paddleaudio"""
    if channels_first:
        data = data.transpose([1, 0])
    scipy.io.wavfile.write(path, sample_rate, data.numpy())
