
from pathlib import Path
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

from paddle import Tensor
from .common import AudioMetaData

from paddlespeech.audio._internal import module_utils  as _mod_utils
from paddlespeech.audio._paddleaudio import get_info_file
from paddlespeech.audio._paddleaudio import get_info_fileobj

#https://github.com/pytorch/audio/blob/main/torchaudio/backend/sox_io_backend.py

def _fail_info(filepath: str, format: Optional[str]) -> AudioMetaData:
    raise RuntimeError("Failed to fetch metadata from {}".format(filepath))


def _fail_info_fileobj(fileobj, format: Optional[str]) -> AudioMetaData:
    raise RuntimeError("Failed to fetch metadata from {}".format(fileobj))


# Note: need to comply TorchScript syntax -- need annotation and no f-string
def _fail_load(
    filepath: str,
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    format: Optional[str] = None,
) -> Tuple[paddle.Tensor, int]:
    raise RuntimeError("Failed to load audio from {}".format(filepath))


def _fail_load_fileobj(fileobj, *args, **kwargs):
    raise RuntimeError(f"Failed to load audio from {fileobj}")

_fallback_info = _fail_info
_fallback_info_fileobj = _fail_info_fileobj
_fallback_load = _fail_load
_fallback_load_filebj = _fail_load_fileobj

def load(
    filepath: Union[str, Path],
    out: Optional[Tensor] = None,
    normalization: Union[bool, float, Callable] = True,
    channels_first: bool = True,
    num_frames: int = 0,
    offset: int = 0,
    filetype: Optional[str] = None,
) -> Tuple[Tensor, int]:
    raise RuntimeError("No audio I/O backend is available.")

def save(filepath: str, src: Tensor, sample_rate: int, precision: int = 16, channels_first: bool = True) -> None:
    raise RuntimeError("No audio I/O backend is available.")

@_mod_utils.requires_sox()
def info(filepath: str, format: Optional[str]) -> None:
    sinfo = paddleaudio._paddleaudio.get_info_file(filepath, format)
    if sinfo is not None:
        return AudioMetaData(*sinfo)
    return _fallback_info(filepath, format)
