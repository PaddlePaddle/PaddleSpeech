#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
# Modified from https://github.com/webdataset/webdataset
#
"""Automatically decode webdataset samples."""
import io
import json
import os
import pickle
import re
import tempfile
from functools import partial

import numpy as np
"""Extensions passed on to the image decoder."""
image_extensions = "jpg jpeg png ppm pgm pbm pnm".split()

################################################################
# handle basic datatypes
################################################################


def paddle_loads(data):
    """Load data using paddle.loads, importing paddle only if needed.

    :param data: data to be decoded
    """
    import io

    import paddle

    stream = io.BytesIO(data)
    return paddle.load(stream)


def tenbin_loads(data):
    from . import tenbin

    return tenbin.decode_buffer(data)


def msgpack_loads(data):
    import msgpack

    return msgpack.unpackb(data)


def npy_loads(data):
    import numpy.lib.format

    stream = io.BytesIO(data)
    return numpy.lib.format.read_array(stream)


def cbor_loads(data):
    import cbor

    return cbor.loads(data)


decoders = {
    "txt": lambda data: data.decode("utf-8"),
    "text": lambda data: data.decode("utf-8"),
    "transcript": lambda data: data.decode("utf-8"),
    "cls": lambda data: int(data),
    "cls2": lambda data: int(data),
    "index": lambda data: int(data),
    "inx": lambda data: int(data),
    "id": lambda data: int(data),
    "json": lambda data: json.loads(data),
    "jsn": lambda data: json.loads(data),
    "pyd": lambda data: pickle.loads(data),
    "pickle": lambda data: pickle.loads(data),
    "pdparams": lambda data: paddle_loads(data),
    "ten": tenbin_loads,
    "tb": tenbin_loads,
    "mp": msgpack_loads,
    "msg": msgpack_loads,
    "npy": npy_loads,
    "npz": lambda data: np.load(io.BytesIO(data)),
    "cbor": cbor_loads,
}


def basichandlers(key, data):
    """Handle basic file decoding.

    This function is usually part of the post= decoders.
    This handles the following forms of decoding:

    - txt -> unicode string
    - cls cls2 class count index inx id -> int
    - json jsn -> JSON decoding
    - pyd pickle -> pickle decoding
    - pdparams -> paddle.loads
    - ten tenbin -> fast tensor loading
    - mp messagepack msg -> messagepack decoding
    - npy -> Python NPY decoding

    :param key: file name extension
    :param data: binary data to be decoded
    """
    extension = re.sub(r".*[.]", "", key)

    if extension in decoders:
        return decoders[extension](data)

    return None


################################################################
# Generic extension handler.
################################################################


def call_extension_handler(key, data, f, extensions):
    """Call the function f with the given data if the key matches the extensions.

    :param key: actual key found in the sample
    :param data: binary data
    :param f: decoder function
    :param extensions: list of matching extensions
    """
    extension = key.lower().split(".")
    for target in extensions:
        target = target.split(".")
        if len(target) > len(extension):
            continue
        if extension[-len(target):] == target:
            return f(data)
    return None


def handle_extension(extensions, f):
    """Return a decoder function for the list of extensions.

    Extensions can be a space separated list of extensions.
    Extensions can contain dots, in which case the corresponding number
    of extension components must be present in the key given to f.
    Comparisons are case insensitive.

    Examples:
    handle_extension("jpg jpeg", my_decode_jpg)  # invoked for any file.jpg
    handle_extension("seg.jpg", special_case_jpg)  # invoked only for file.seg.jpg
    """
    extensions = extensions.lower().split()
    return partial(call_extension_handler, f=f, extensions=extensions)


################################################################
# handle images
################################################################

imagespecs = {
    "l8": ("numpy", "uint8", "l"),
    "rgb8": ("numpy", "uint8", "rgb"),
    "rgba8": ("numpy", "uint8", "rgba"),
    "l": ("numpy", "float", "l"),
    "rgb": ("numpy", "float", "rgb"),
    "rgba": ("numpy", "float", "rgba"),
    "paddlel8": ("paddle", "uint8", "l"),
    "paddlergb8": ("paddle", "uint8", "rgb"),
    "paddlergba8": ("paddle", "uint8", "rgba"),
    "paddlel": ("paddle", "float", "l"),
    "paddlergb": ("paddle", "float", "rgb"),
    "paddle": ("paddle", "float", "rgb"),
    "paddlergba": ("paddle", "float", "rgba"),
    "pill": ("pil", None, "l"),
    "pil": ("pil", None, "rgb"),
    "pilrgb": ("pil", None, "rgb"),
    "pilrgba": ("pil", None, "rgba"),
}


class ImageHandler:
    """Decode image data using the given `imagespec`.

    The `imagespec` specifies whether the image is decoded
    to numpy/paddle/pi, decoded to uint8/float, and decoded
    to l/rgb/rgba:

    - l8: numpy uint8 l
    - rgb8: numpy uint8 rgb
    - rgba8: numpy uint8 rgba
    - l: numpy float l
    - rgb: numpy float rgb
    - rgba: numpy float rgba
    - paddlel8: paddle uint8 l
    - paddlergb8: paddle uint8 rgb
    - paddlergba8: paddle uint8 rgba
    - paddlel: paddle float l
    - paddlergb: paddle float rgb
    - paddle: paddle float rgb
    - paddlergba: paddle float rgba
    - pill: pil None l
    - pil: pil None rgb
    - pilrgb: pil None rgb
    - pilrgba: pil None rgba

    """

    def __init__(self, imagespec, extensions=image_extensions):
        """Create an image handler.

        :param imagespec: short string indicating the type of decoding
        :param extensions: list of extensions the image handler is invoked for
        """
        if imagespec not in list(imagespecs.keys()):
            raise ValueError("Unknown imagespec: %s" % imagespec)
        self.imagespec = imagespec.lower()
        self.extensions = extensions

    def __call__(self, key, data):
        """Perform image decoding.

        :param key: file name extension
        :param data: binary data
        """
        import PIL.Image

        extension = re.sub(r".*[.]", "", key)
        if extension.lower() not in self.extensions:
            return None
        imagespec = self.imagespec
        atype, etype, mode = imagespecs[imagespec]
        with io.BytesIO(data) as stream:
            img = PIL.Image.open(stream)
            img.load()
            img = img.convert(mode.upper())
        if atype == "pil":
            return img
        elif atype == "numpy":
            result = np.asarray(img)
            if result.dtype != np.uint8:
                raise ValueError("ImageHandler: numpy image must be uint8")
            if etype == "uint8":
                return result
            else:
                return result.astype("f") / 255.0
        elif atype == "paddle":
            import paddle

            result = np.asarray(img)
            if result.dtype != np.uint8:
                raise ValueError("ImageHandler: paddle image must be uint8")
            if etype == "uint8":
                result = np.array(result.transpose(2, 0, 1))
                return paddle.tensor(result)
            else:
                result = np.array(result.transpose(2, 0, 1))
                return paddle.tensor(result) / 255.0
        return None


def imagehandler(imagespec, extensions=image_extensions):
    """Create an image handler.

    This is just a lower case alias for ImageHander.

    :param imagespec: textual image spec
    :param extensions: list of extensions the handler should be applied for
    """
    return ImageHandler(imagespec, extensions)


################################################################
# torch video
################################################################
'''
def torch_video(key, data):
    """Decode video using the torchvideo library.

    :param key: file name extension
    :param data: data to be decoded
    """
    extension = re.sub(r".*[.]", "", key)
    if extension not in "mp4 ogv mjpeg avi mov h264 mpg webm wmv".split():
        return None

    import torchvision.io

    with tempfile.TemporaryDirectory() as dirname:
        fname = os.path.join(dirname, f"file.{extension}")
        with open(fname, "wb") as stream:
            stream.write(data)
        return torchvision.io.read_video(fname, pts_unit="sec")
'''

################################################################
# paddlespeech.audio
################################################################


def paddle_audio(key, data):
    """Decode audio using the paddlespeech.audio library.

    :param key: file name extension
    :param data: data to be decoded
    """
    extension = re.sub(r".*[.]", "", key)
    if extension not in ["flac", "mp3", "sox", "wav", "m4a", "ogg", "wma"]:
        return None

    import paddlespeech.audio

    with tempfile.TemporaryDirectory() as dirname:
        fname = os.path.join(dirname, f"file.{extension}")
        with open(fname, "wb") as stream:
            stream.write(data)
        return paddleaudio.backends.soundfile_load(fname)


################################################################
# special class for continuing decoding
################################################################


class Continue:
    """Special class for continuing decoding.

    This is mostly used for decompression, as in:

        def decompressor(key, data):
            if key.endswith(".gz"):
                return Continue(key[:-3], decompress(data))
            return None
    """

    def __init__(self, key, data):
        """__init__.

        :param key:
        :param data:
        """
        self.key, self.data = key, data


def gzfilter(key, data):
    """Decode .gz files.

    This decodes compressed files and the continues decoding.

    :param key: file name extension
    :param data: binary data
    """
    import gzip

    if not key.endswith(".gz"):
        return None
    decompressed = gzip.open(io.BytesIO(data)).read()
    return Continue(key[:-3], decompressed)


################################################################
# decode entire training amples
################################################################

default_pre_handlers = [gzfilter]
default_post_handlers = [basichandlers]


class Decoder:
    """Decode samples using a list of handlers.

    For each key/data item, this iterates through the list of
    handlers until some handler returns something other than None.
    """

    def __init__(self, handlers, pre=None, post=None, only=None, partial=False):
        """Create a Decoder.

        :param handlers: main list of handlers
        :param pre: handlers called before the main list (.gz handler by default)
        :param post: handlers called after the main list (default handlers by default)
        :param only: a list of extensions; when give, only ignores files with those extensions
        :param partial: allow partial decoding (i.e., don't decode fields that aren't of type bytes)
        """
        if isinstance(only, str):
            only = only.split()
        self.only = only if only is None else set(only)
        if pre is None:
            pre = default_pre_handlers
        if post is None:
            post = default_post_handlers
        assert all(callable(h)
                   for h in handlers), f"one of {handlers} not callable"
        assert all(callable(h) for h in pre), f"one of {pre} not callable"
        assert all(callable(h) for h in post), f"one of {post} not callable"
        self.handlers = pre + handlers + post
        self.partial = partial

    def decode1(self, key, data):
        """Decode a single field of a sample.

        :param key: file name extension
        :param data: binary data
        """
        key = "." + key
        for f in self.handlers:
            result = f(key, data)
            if isinstance(result, Continue):
                key, data = result.key, result.data
                continue
            if result is not None:
                return result
        return data

    def decode(self, sample):
        """Decode an entire sample.

        :param sample: the sample, a dictionary of key value pairs
        """
        result = {}
        assert isinstance(sample, dict), sample
        for k, v in list(sample.items()):
            if k[0] == "_":
                if isinstance(v, bytes):
                    v = v.decode("utf-8")
                result[k] = v
                continue
            if self.only is not None and k not in self.only:
                result[k] = v
                continue
            assert v is not None
            if self.partial:
                if isinstance(v, bytes):
                    result[k] = self.decode1(k, v)
                else:
                    result[k] = v
            else:
                assert isinstance(v, bytes)
                result[k] = self.decode1(k, v)
        return result

    def __call__(self, sample):
        """Decode an entire sample.

        :param sample: the sample
        """
        assert isinstance(sample, dict), (len(sample), sample)
        return self.decode(sample)
