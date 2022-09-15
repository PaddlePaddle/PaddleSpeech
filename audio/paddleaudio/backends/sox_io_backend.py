from pathlib import Path
from typing import Callable
from typing import Optional, Tuple, Union

import paddle
import paddleaudio
from paddle import Tensor
from .common import AudioInfo
import os

from paddleaudio._internal import module_utils  as _mod_utils

#https://github.com/pytorch/audio/blob/main/torchaudio/backend/sox_io_backend.py

def _fail_info(filepath: str, format: Optional[str]) -> AudioInfo:
    raise RuntimeError("Failed to fetch metadata from {}".format(filepath))


def _fail_info_fileobj(fileobj, format: Optional[str]) -> AudioInfo:
    raise RuntimeError("Failed to fetch metadata from {}".format(fileobj))


# Note: need to comply TorchScript syntax -- need annotation and no f-string
def _fail_load(
    filepath: str,
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    format: Optional[str] = None,
) -> Tuple[Tensor, int]:
    raise RuntimeError("Failed to load audio from {}".format(filepath))


def _fail_load_fileobj(fileobj, *args, **kwargs):
    raise RuntimeError(f"Failed to load audio from {fileobj}")

_fallback_info = _fail_info
_fallback_info_fileobj = _fail_info_fileobj
_fallback_load = _fail_load
_fallback_load_filebj = _fail_load_fileobj

@_mod_utils.requires_sox()
def load(
        filepath: str,
        frame_offset: int = 0,
        num_frames: int=-1,
        normalize: bool = True,
        channels_first: bool = True,
        format: Optional[str]=None, ) -> Tuple[Tensor, int]:
    if hasattr(filepath, "read"):
        ret = paddleaudio._paddleaudio.load_audio_fileobj(
            filepath, frame_offset, num_frames, normalize, channels_first, format
        )
        if ret is not None:
            audio_tensor = paddle.to_tensor(ret[0])
            return (audio_tensor, ret[1])
        return _fallback_load_fileobj(filepath, frame_offset, num_frames, normalize, channels_first, format)
    filepath = os.fspath(filepath)
    ret = paddleaudio._paddleaudio.sox_io_load_audio_file(
        filepath, frame_offset, num_frames, normalize, channels_first, format
    )
    if ret is not None:
        audio_tensor = paddle.to_tensor(ret[0])
        return (audio_tensor, ret[1])
    return _fallback_load(filepath, frame_offset, num_frames, normalize, channels_first, format)


@_mod_utils.requires_sox()
def save(filepath: str,
    src: Tensor,
    sample_rate: int,
    channels_first: bool = True,
    compression: Optional[float] = None,
    format: Optional[str] = None,
    encoding: Optional[str] = None,
    bits_per_sample: Optional[int] = None,
):
    src_arr = src.numpy()
    if hasattr(filepath, "write"):
        paddleaudio._paddleaudio.save_audio_fileobj(
            filepath, src_arr, sample_rate, channels_first, compression, format, encoding, bits_per_sample
        )
        return
    filepath = os.fspath(filepath)
    paddleaudio._paddleaudio.sox_io_save_audio_file(
        filepath, src_arr, sample_rate, channels_first, compression, format, encoding, bits_per_sample
    )

@_mod_utils.requires_sox()
def info(filepath: str, format: Optional[str] = None,) -> AudioInfo:
    if hasattr(filepath, "read"):
        sinfo = paddleaudio._paddleaudio.get_info_fileobj(filepath, format)
        if sinfo is not None:
            return AudioInfo(*sinfo)
        return _fallback_info_fileobj(filepath, format)
    filepath = os.fspath(filepath)
    sinfo = paddleaudio._paddleaudio.get_info_file(filepath, format)
    if sinfo is not None:
        return AudioInfo(*sinfo)
    return _fallback_info(filepath, format)
