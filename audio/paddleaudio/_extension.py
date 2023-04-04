import contextlib
import ctypes
import os
import sys
import types
import warnings
from pathlib import Path

from ._internal import module_utils as _mod_utils  # noqa: F401

# Query `hasattr` only once.
_SET_GLOBAL_FLAGS = hasattr(sys, 'getdlopenflags') and hasattr(
    sys, 'setdlopenflags')


@contextlib.contextmanager
def dl_open_guard():
    """
    # https://manpages.debian.org/bullseye/manpages-dev/dlopen.3.en.html
    Context manager to set the RTLD_GLOBAL dynamic linker flag while we open a
    shared library to load custom operators.
    """
    if _SET_GLOBAL_FLAGS:
        old_flags = sys.getdlopenflags()
        sys.setdlopenflags(old_flags | ctypes.RTLD_GLOBAL)
    yield
    if _SET_GLOBAL_FLAGS:
        sys.setdlopenflags(old_flags)


def resolve_library_path(path: str) -> str:
    return os.path.realpath(path)


class _Ops(types.ModuleType):
    #__file__ = '_ops.py'

    def __init__(self):
        super(_Ops, self).__init__('paddleaudio.ops')
        self.loaded_libraries = set()

    def load_library(self, path):
        """
        Loads a shared library from the given path into the current process.
        This allows dynamically loading custom operators. For this, 
        you should compile your operator and 
        the static registration code into a shared library object, and then
        call ``paddleaudio.ops.load_library('path/to/libcustom.so')`` to load the
        shared object.
        After the library is loaded, it is added to the
        ``paddleaudio.ops.loaded_libraries`` attribute, a set that may be inspected
        for the paths of all libraries loaded using this function.
        Args:
            path (str): A path to a shared library to load.
        """
        path = resolve_library_path(path)
        with dl_open_guard():
            # https://docs.python.org/3/library/ctypes.html?highlight=ctypes#loading-shared-libraries
            # Import the shared library into the process, thus running its
            # static (global) initialization code in order to register custom
            # operators with the JIT.
            ctypes.CDLL(path)
        self.loaded_libraries.add(path)


_LIB_DIR = Path(__file__).parent / "lib"


def _get_lib_path(lib: str):
    suffix = "pyd" if os.name == "nt" else "so"
    path = _LIB_DIR / f"{lib}.{suffix}"
    return path


def _load_lib(lib: str) -> bool:
    """Load extension module
    Note:
        In case `paddleaudio` is deployed with `pex` format, the library file
        is not in a standard location.
        In this case, we expect that `libpaddlleaudio` is available somewhere
        in the search path of dynamic loading mechanism, so that importing
        `_paddlleaudio` will have library loader find and load `libpaddlleaudio`.
        This is the reason why the function should not raising an error when the library
        file is not found.
    Returns:
        bool:
            True if the library file is found AND the library loaded without failure.
            False if the library file is not found (like in the case where paddlleaudio
            is deployed with pex format, thus the shared library file is
            in a non-standard location.).
            If the library file is found but there is an issue loading the library,
            (such as missing dependency) then this function raises the exception as-is.
    Raises:
        Exception:
            If the library file is found, but there is an issue loading the library file,
            (when underlying `ctype.DLL` throws an exception), this function will pass
            the exception as-is, instead of catching it and returning bool.
            The expected case is `OSError` thrown by `ctype.DLL` when a dynamic dependency
            is not found.
            This behavior was chosen because the expected failure case is not recoverable.
            If a dependency is missing, then users have to install it.
    """
    path = _get_lib_path(lib)
    if not path.exists():
        warnings.warn("lib path is not exists:" + str(path))
        return False
    ops.load_library(path)
    return True


_FFMPEG_INITIALIZED = False


def _init_ffmpeg():
    global _FFMPEG_INITIALIZED
    if _FFMPEG_INITIALIZED:
        return

    if not paddleaudio._paddlleaudio.is_ffmpeg_available():
        raise RuntimeError(
            "paddlleaudio is not compiled with FFmpeg integration. Please set USE_FFMPEG=1 when compiling paddlleaudio."
        )

    try:
        _load_lib("libpaddlleaudio_ffmpeg")
    except OSError as err:
        raise ImportError(
            "FFmpeg libraries are not found. Please install FFmpeg.") from err

    import paddllespeech.audio._paddlleaudio_ffmpeg  # noqa

    paddleaudio._paddlleaudio.ffmpeg_init()
    if paddleaudio._paddlleaudio.ffmpeg_get_log_level() > 8:
        paddleaudio._paddlleaudio.ffmpeg_set_log_level(8)

    _FFMPEG_INITIALIZED = True


def _init_extension():
    if not _mod_utils.is_module_available("paddleaudio._paddleaudio"):
        warnings.warn(
            "paddleaudio C++ extension is not available. sox_io, sox_effect, kaldi raw feature is not supported!!!"
        )
        return

    _load_lib("libpaddleaudio")
    # This import is for initializing the methods registered via PyBind11
    # This has to happen after the base library is loaded
    try:
        from paddleaudio import _paddleaudio  # noqa
    except Exception:
        warnings.warn(
            "paddleaudio C++ extension is not available. sox_io, sox_effect, kaldi raw feature is not supported!!!"
        )
        return

    # Because this part is executed as part of `import torchaudio`, we ignore the
    # initialization failure.
    # If the FFmpeg integration is not properly initialized, then detailed error
    # will be raised when client code attempts to import the dedicated feature.
    try:
        _init_ffmpeg()
    except Exception:
        pass


ops = _Ops()

_init_extension()
