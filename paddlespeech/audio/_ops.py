import contextlib
import ctypes
import os
import sys
import types

# Query `hasattr` only once.
_SET_GLOBAL_FLAGS = hasattr(sys, 'getdlopenflags') and hasattr(sys,
                                                               'setdlopenflags')


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
    __file__ = '_ops.py'

    def __init__(self):
        super(_Ops, self).__init__('paddlespeech.ops')
        self.loaded_libraries = set()

    def load_library(self, path):
        """
        Loads a shared library from the given path into the current process.
        This allows dynamically loading custom operators. For this, 
        you should compile your operator and 
        the static registration code into a shared library object, and then
        call ``paddlespeech.ops.load_library('path/to/libcustom.so')`` to load the
        shared object.
        After the library is loaded, it is added to the
        ``paddlespeech.ops.loaded_libraries`` attribute, a set that may be inspected
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


# The ops "namespace"
ops = _Ops()
