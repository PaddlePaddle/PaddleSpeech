import functools
import os.path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

#code is from:https://github.com/pytorch/audio/blob/main/test/torchaudio_unittest/common_utils/case_utils.py

import paddle
from paddlespeech.audio._internal.module_utils import (
    is_kaldi_available,
    is_module_available,
    is_sox_available,
)

def name_func(func, _, params):
    return f'{func.__name__}_{"_".join(str(arg) for arg in params.args)}'

class TempDirMixin:
    """Mixin to provide easy access to temp dir"""

    temp_dir_ = None

    @classmethod
    def get_base_temp_dir(cls):
        # If PADDLEAUDIO_TEST_TEMP_DIR is set, use it instead of temporary directory.
        # this is handy for debugging.
        key = "PADDLEAUDIO_TEST_TEMP_DIR"
        if key in os.environ:
            return os.environ[key]
        if cls.temp_dir_ is None:
            cls.temp_dir_ = tempfile.TemporaryDirectory()
        return cls.temp_dir_.name

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir_ is not None:
            try:
                cls.temp_dir_.cleanup()
                cls.temp_dir_ = None
            except PermissionError:
                # On Windows there is a know issue with `shutil.rmtree`,
                # which fails intermittenly.
                #
                # https://github.com/python/cpython/issues/74168
                #
                # We observed this on CircleCI, where Windows job raises
                # PermissionError.
                #
                # Following the above thread, we ignore it.
                pass
        super().tearDownClass()

    def get_temp_path(self, *paths):
        temp_dir = os.path.join(self.get_base_temp_dir(), self.id())
        path = os.path.join(temp_dir, *paths)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
