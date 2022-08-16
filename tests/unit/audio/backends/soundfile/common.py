import itertools
from unittest import skipIf

from parameterized import parameterized
from paddlespeech.audio._internal.module_utils import is_module_available


def name_func(func, _, params):
    return f'{func.__name__}_{"_".join(str(arg) for arg in params.args)}'


def dtype2subtype(dtype):
    return {
        "float64": "DOUBLE",
        "float32": "FLOAT",
        "int32": "PCM_32",
        "int16": "PCM_16",
        "uint8": "PCM_U8",
        "int8": "PCM_S8",
    }[dtype]


def skipIfFormatNotSupported(fmt):
    fmts = []
    if is_module_available("soundfile"):
        import soundfile

        fmts = soundfile.available_formats()
        return skipIf(fmt not in fmts, f'"{fmt}" is not supported by soundfile')
    return skipIf(True, '"soundfile" not available.')


def parameterize(*params):
    return parameterized.expand(list(itertools.product(*params)), name_func=name_func)


def fetch_wav_subtype(dtype, encoding, bits_per_sample):
    subtype = {
        (None, None): dtype2subtype(dtype),
        (None, 8): "PCM_U8",
        ("PCM_U", None): "PCM_U8",
        ("PCM_U", 8): "PCM_U8",
        ("PCM_S", None): "PCM_32",
        ("PCM_S", 16): "PCM_16",
        ("PCM_S", 32): "PCM_32",
        ("PCM_F", None): "FLOAT",
        ("PCM_F", 32): "FLOAT",
        ("PCM_F", 64): "DOUBLE",
        ("ULAW", None): "ULAW",
        ("ULAW", 8): "ULAW",
        ("ALAW", None): "ALAW",
        ("ALAW", 8): "ALAW",
    }.get((encoding, bits_per_sample))
    if subtype:
        return subtype
    raise ValueError(f"wav does not support ({encoding}, {bits_per_sample}).")

