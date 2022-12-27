"""Defines utilities for switching audio backends"""
#code is from: https://github.com/pytorch/audio/blob/main/torchaudio/backend/utils.py
import warnings
from typing import List
from typing import Optional

import paddleaudio
from paddleaudio._internal import module_utils as _mod_utils

from . import no_backend
from . import soundfile_backend
from . import sox_io_backend

__all__ = [
    "list_audio_backends",
    "get_audio_backend",
    "set_audio_backend",
]


def list_audio_backends() -> List[str]:
    """List available backends

    Returns:
        List[str]: The list of available backends.
    """
    backends = []
    if _mod_utils.is_module_available("soundfile"):
        backends.append("soundfile")
    if _mod_utils.is_sox_available():
        backends.append("sox_io")
    return backends


def set_audio_backend(backend: Optional[str]):
    """Set the backend for I/O operation

    Args:
        backend (str or None): Name of the backend.
            One of ``"sox_io"`` or ``"soundfile"`` based on availability
            of the system. If ``None`` is provided the  current backend is unassigned.
    """
    if backend is not None and backend not in list_audio_backends():
        raise RuntimeError(f'Backend "{backend}" is not one of '
                           f"available backends: {list_audio_backends()}.")

    if backend is None:
        module = no_backend
    elif backend == "sox_io":
        module = sox_io_backend
    elif backend == "soundfile":
        module = soundfile_backend
    else:
        raise NotImplementedError(f'Unexpected backend "{backend}"')

    for func in ["save", "load", "info"]:
        setattr(paddleaudio, func, getattr(module, func))


def _init_audio_backend():
    backends = list_audio_backends()
    if "soundfile" in backends:
        set_audio_backend("soundfile")
    elif "sox_io" in backends:
        set_audio_backend("sox_io")
    else:
        warnings.warn("No audio backend is available.")
        set_audio_backend(None)


def get_audio_backend() -> Optional[str]:
    """Get the name of the current backend

    Returns:
        Optional[str]: The name of the current backend or ``None`` if no backend is assigned.
    """
    if paddleaudio.load == no_backend.load:
        return None
    if paddleaudio.load == sox_io_backend.load:
        return "sox_io"
    if paddleaudio.load == soundfile_backend.load:
        return "soundfile"
    raise ValueError("Unknown backend.")
