import json
import os.path

import paddle
from parameterized import param
#code is from:https://github.com/pytorch/audio/blob/main/test/torchaudio_unittest/common_utils/data_utils.py with modification.

_TEST_DIR_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))


def get_asset_path(*paths):
    """Return full path of a test asset"""
    return os.path.join(_TEST_DIR_PATH, "assets", *paths)


def load_params(*paths):
    with open(get_asset_path(*paths), "r") as file:
        return [param(json.loads(line)) for line in file]


def load_effects_params(*paths):
    params = []
    with open(*paths, "r") as file:
        for line in file:
            data = json.loads(line)
            for effect in data["effects"]:
                for i, arg in enumerate(effect):
                    if arg.startswith("<ASSET_DIR>"):
                        effect[i] = arg.replace("<ASSET_DIR>", get_asset_path())
            params.append(param(data))
    return params


def convert_tensor_encoding(
        tensor: paddle.tensor,
        dtype: paddle.dtype, ):
    """Convert input tensor with values between -1 and 1 to integer encoding
    Args:
        tensor: input tensor, assumed between -1 and 1
        dtype: desired output tensor dtype
    Returns:
        Tensor: shape of (n_channels, sample_rate * duration)
    """
    if dtype == paddle.int32:
        tensor *= (tensor > 0) * 2147483647 + (tensor < 0) * 2147483648
    if dtype == paddle.int16:
        tensor *= (tensor > 0) * 32767 + (tensor < 0) * 32768
    if dtype == paddle.uint8:
        tensor *= (tensor > 0) * 127 + (tensor < 0) * 128
        tensor += 128
    tensor = paddle.to_tensor(tensor, dtype)
    return tensor


#def get_whitenoise(
#*,
#sample_rate: int = 16000,
#duration: float = 1,  # seconds
#n_channels: int = 1,
#seed: int = 0,
#dtype: Union[str, paddle.dtype] = "float32",
#device: Union[str, paddle.device] = "cpu",
#channels_first=True,
#scale_factor: float = 1,
#):
#"""Generate pseudo audio data with whitenoise
#Args:
#sample_rate: Sampling rate
#duration: Length of the resulting Tensor in seconds.
#n_channels: Number of channels
#seed: Seed value used for random number generation.
#Note that this function does not modify global random generator state.
#dtype: Torch dtype
#device: device
#channels_first: whether first dimension is n_channels
#scale_factor: scale the Tensor before clamping and quantization
#Returns:
#Tensor: shape of (n_channels, sample_rate * duration)
#"""
#if isinstance(dtype, str):
#dtype = getattr(paddle, dtype)
#if dtype not in [paddle.float64, paddle.float32, paddle.int32, paddle.int16, paddle.uint8]:
#raise NotImplementedError(f"dtype {dtype} is not supported.")
## According to the doc, folking rng on all CUDA devices is slow when there are many CUDA devices,
## so we only fork on CPU, generate values and move the data to the given device
#with paddle.random.fork_rng([]):
#paddle.random.manual_seed(seed)
#tensor = paddle.randn([n_channels, int(sample_rate * duration)], dtype=paddle.float32, device="cpu")
#tensor /= 2.0
#tensor *= scale_factor
#tensor.clamp_(-1.0, 1.0)
#if not channels_first:
#tensor = tensor.t()

#tensor = tensor.to(device)

#return convert_tensor_encoding(tensor, dtype)


def get_sinusoid(
        *,
        frequency: float=300,
        sample_rate: int=16000,
        duration: float=1,  # seconds
        n_channels: int=1,
        dtype: str="float32",
        device: str="cpu",
        channels_first: bool=True, ):
    """Generate pseudo audio data with sine wave.

    Args:
        frequency: Frequency of sine wave
        sample_rate: Sampling rate
        duration: Length of the resulting Tensor in seconds.
        n_channels: Number of channels
        dtype: Torch dtype
        device: device

    Returns:
        Tensor: shape of (n_channels, sample_rate * duration)
    """
    if isinstance(dtype, str):
        dtype = getattr(paddle, dtype)
    pie2 = 2 * 3.141592653589793
    end = pie2 * frequency * duration
    num_frames = int(sample_rate * duration)
    # Randomize the initial phase. (except the first channel)
    theta0 = pie2 * paddle.randn([n_channels, 1], dtype=paddle.float32)
    theta0[0, :] = 0
    theta = paddle.linspace(0, end, num_frames, dtype=paddle.float32)
    theta = theta0 + theta
    tensor = paddle.sin(theta)
    if not channels_first:
        tensor = paddle.t(tensor)
    return convert_tensor_encoding(tensor, dtype)
