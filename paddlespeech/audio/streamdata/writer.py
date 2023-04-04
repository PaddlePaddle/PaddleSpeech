#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
# Modified from https://github.com/webdataset/webdataset
#
"""Classes and functions for writing tar files and WebDataset files."""
import io
import json
import pickle
import re
import tarfile
import time
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

import numpy as np

from . import gopen


def imageencoder(image: Any, format: str = "PNG"):  # skipcq: PYL-W0622
    """Compress an image using PIL and return it as a string.

    Can handle float or uint8 images.

    :param image: ndarray representing an image
    :param format: compression format (PNG, JPEG, PPM)

    """
    import PIL

    assert isinstance(image, (PIL.Image.Image, np.ndarray)), type(image)

    if isinstance(image, np.ndarray):
        if image.dtype in [np.dtype("f"), np.dtype("d")]:
            if not (np.amin(image) > -0.001 and np.amax(image) < 1.001):
                raise ValueError(
                    f"image values out of range {np.amin(image)} {np.amax(image)}"
                )
            image = np.clip(image, 0.0, 1.0)
            image = np.array(image * 255.0, "uint8")
        assert image.ndim in [2, 3]
        if image.ndim == 3:
            assert image.shape[2] in [1, 3]
        image = PIL.Image.fromarray(image)
    if format.upper() == "JPG":
        format = "JPEG"
    elif format.upper() in ["IMG", "IMAGE"]:
        format = "PPM"
    if format == "JPEG":
        opts = dict(quality=100)
    else:
        opts = {}
    with io.BytesIO() as result:
        image.save(result, format=format, **opts)
        return result.getvalue()


def bytestr(data: Any):
    """Convert data into a bytestring.

    Uses str and ASCII encoding for data that isn't already in string format.

    :param data: data
    """
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode("ascii")
    return str(data).encode("ascii")


def paddle_dumps(data: Any):
    """Dump data into a bytestring using paddle.dumps.

    This delays importing paddle until needed.

    :param data: data to be dumped
    """
    import io

    import paddle

    stream = io.BytesIO()
    paddle.save(data, stream)
    return stream.getvalue()


def numpy_dumps(data: np.ndarray):
    """Dump data into a bytestring using numpy npy format.

    :param data: data to be dumped
    """
    import io

    import numpy.lib.format

    stream = io.BytesIO()
    numpy.lib.format.write_array(stream, data)
    return stream.getvalue()


def numpy_npz_dumps(data: np.ndarray):
    """Dump data into a bytestring using numpy npz format.

    :param data: data to be dumped
    """
    import io

    stream = io.BytesIO()
    np.savez_compressed(stream, **data)
    return stream.getvalue()


def tenbin_dumps(x):
    from . import tenbin

    if isinstance(x, list):
        return memoryview(tenbin.encode_buffer(x))
    else:
        return memoryview(tenbin.encode_buffer([x]))


def cbor_dumps(x):
    import cbor

    return cbor.dumps(x)


def mp_dumps(x):
    import msgpack

    return msgpack.packb(x)


def add_handlers(d, keys, value):
    if isinstance(keys, str):
        keys = keys.split()
    for k in keys:
        d[k] = value


def make_handlers():
    """Create a list of handlers for encoding data."""
    handlers = {}
    add_handlers(handlers, "cls cls2 class count index inx id",
                 lambda x: str(x).encode("ascii"))
    add_handlers(handlers, "txt text transcript", lambda x: x.encode("utf-8"))
    add_handlers(handlers, "html htm", lambda x: x.encode("utf-8"))
    add_handlers(handlers, "pyd pickle", pickle.dumps)
    add_handlers(handlers, "pdparams", paddle_dumps)
    add_handlers(handlers, "npy", numpy_dumps)
    add_handlers(handlers, "npz", numpy_npz_dumps)
    add_handlers(handlers, "ten tenbin tb", tenbin_dumps)
    add_handlers(handlers, "json jsn", lambda x: json.dumps(x).encode("utf-8"))
    add_handlers(handlers, "mp msgpack msg", mp_dumps)
    add_handlers(handlers, "cbor", cbor_dumps)
    add_handlers(handlers, "jpg jpeg img image",
                 lambda data: imageencoder(data, "jpg"))
    add_handlers(handlers, "png", lambda data: imageencoder(data, "png"))
    add_handlers(handlers, "pbm", lambda data: imageencoder(data, "pbm"))
    add_handlers(handlers, "pgm", lambda data: imageencoder(data, "pgm"))
    add_handlers(handlers, "ppm", lambda data: imageencoder(data, "ppm"))
    return handlers


default_handlers = make_handlers()


def encode_based_on_extension1(data: Any, tname: str, handlers: dict):
    """Encode data based on its extension and a dict of handlers.

    :param data: data
    :param tname: file extension
    :param handlers: handlers
    """
    if tname[0] == "_":
        if not isinstance(data, str):
            raise ValueError("the values of metadata must be of string type")
        return data
    extension = re.sub(r".*\.", "", tname).lower()
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode("utf-8")
    handler = handlers.get(extension)
    if handler is None:
        raise ValueError(f"no handler found for {extension}")
    return handler(data)


def encode_based_on_extension(sample: dict, handlers: dict):
    """Encode an entire sample with a collection of handlers.

    :param sample: data sample (a dict)
    :param handlers: handlers for encoding
    """
    return {
        k: encode_based_on_extension1(v, k, handlers)
        for k, v in list(sample.items())
    }


def make_encoder(spec: Union[bool, str, dict, Callable]):
    """Make an encoder function from a specification.

    :param spec: specification
    """
    if spec is False or spec is None:

        def encoder(x):
            """Do not encode at all."""
            return x

    elif callable(spec):
        encoder = spec
    elif isinstance(spec, dict):

        def f(sample):
            """Encode based on extension."""
            return encode_based_on_extension(sample, spec)

        encoder = f

    elif spec is True:
        handlers = default_handlers

        def g(sample):
            """Encode based on extension."""
            return encode_based_on_extension(sample, handlers)

        encoder = g

    else:
        raise ValueError(f"{spec}: unknown decoder spec")
    if not callable(encoder):
        raise ValueError(f"{spec} did not yield a callable encoder")
    return encoder


class TarWriter:
    """A class for writing dictionaries to tar files.

    :param fileobj: fileobj: file name for tar file (.tgz/.tar) or open file descriptor
    :param encoder: sample encoding (Default value = True)
    :param compress:  (Default value = None)

    `True` will use an encoder that behaves similar to the automatic
    decoder for `Dataset`. `False` disables encoding and expects byte strings
    (except for metadata, which must be strings). The `encoder` argument can
    also be a `callable`, or a dictionary mapping extensions to encoders.

    The following code will add two file to the tar archive: `a/b.png` and
    `a/b.output.png`.

    ```Python
        tarwriter = TarWriter(stream)
        image = imread("b.jpg")
        image2 = imread("b.out.jpg")
        sample = {"__key__": "a/b", "png": image, "output.png": image2}
        tarwriter.write(sample)
    ```
    """
    def __init__(
        self,
        fileobj,
        user: str = "bigdata",
        group: str = "bigdata",
        mode: int = 0o0444,
        compress: Optional[bool] = None,
        encoder: Union[None, bool, Callable] = True,
        keep_meta: bool = False,
    ):
        """Create a tar writer.

        :param fileobj: stream to write data to
        :param user: user for tar files
        :param group: group for tar files
        :param mode: mode for tar files
        :param compress: desired compression
        :param encoder: encoder function
        :param keep_meta: keep metadata (entries starting with "_")
        """
        if isinstance(fileobj, str):
            if compress is False:
                tarmode = "w|"
            elif compress is True:
                tarmode = "w|gz"
            else:
                tarmode = "w|gz" if fileobj.endswith("gz") else "w|"
            fileobj = gopen.gopen(fileobj, "wb")
            self.own_fileobj = fileobj
        else:
            tarmode = "w|gz" if compress is True else "w|"
            self.own_fileobj = None
        self.encoder = make_encoder(encoder)
        self.keep_meta = keep_meta
        self.stream = fileobj
        self.tarstream = tarfile.open(fileobj=fileobj, mode=tarmode)

        self.user = user
        self.group = group
        self.mode = mode
        self.compress = compress

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        self.close()

    def close(self):
        """Close the tar file."""
        self.tarstream.close()
        if self.own_fileobj is not None:
            self.own_fileobj.close()
            self.own_fileobj = None

    def write(self, obj):
        """Write a dictionary to the tar file.

        :param obj: dictionary of objects to be stored
        :returns: size of the entry

        """
        total = 0
        obj = self.encoder(obj)
        if "__key__" not in obj:
            raise ValueError("object must contain a __key__")
        for k, v in list(obj.items()):
            if k[0] == "_":
                continue
            if not isinstance(v, (bytes, bytearray, memoryview)):
                raise ValueError(
                    f"{k} doesn't map to a bytes after encoding ({type(v)})")
        key = obj["__key__"]
        for k in sorted(obj.keys()):
            if k == "__key__":
                continue
            if not self.keep_meta and k[0] == "_":
                continue
            v = obj[k]
            if isinstance(v, str):
                v = v.encode("utf-8")
            now = time.time()
            ti = tarfile.TarInfo(key + "." + k)
            ti.size = len(v)
            ti.mtime = now
            ti.mode = self.mode
            ti.uname = self.user
            ti.gname = self.group
            if not isinstance(v, (bytes, bytearray, memoryview)):
                raise ValueError(
                    f"converter didn't yield bytes: {k}, {type(v)}")
            stream = io.BytesIO(v)
            self.tarstream.addfile(ti, stream)
            total += ti.size
        return total


class ShardWriter:
    """Like TarWriter but splits into multiple shards."""
    def __init__(
        self,
        pattern: str,
        maxcount: int = 100000,
        maxsize: float = 3e9,
        post: Optional[Callable] = None,
        start_shard: int = 0,
        **kw,
    ):
        """Create a ShardWriter.

        :param pattern: output file pattern
        :param maxcount: maximum number of records per shard (Default value = 100000)
        :param maxsize: maximum size of each shard (Default value = 3e9)
        :param kw: other options passed to TarWriter
        """
        self.verbose = 1
        self.kw = kw
        self.maxcount = maxcount
        self.maxsize = maxsize
        self.post = post

        self.tarstream = None
        self.shard = start_shard
        self.pattern = pattern
        self.total = 0
        self.count = 0
        self.size = 0
        self.fname = None
        self.next_stream()

    def next_stream(self):
        """Close the current stream and move to the next."""
        self.finish()
        self.fname = self.pattern % self.shard
        if self.verbose:
            print(
                "# writing",
                self.fname,
                self.count,
                "%.1f GB" % (self.size / 1e9),
                self.total,
            )
        self.shard += 1
        stream = open(self.fname, "wb")
        self.tarstream = TarWriter(stream, **self.kw)
        self.count = 0
        self.size = 0

    def write(self, obj):
        """Write a sample.

        :param obj: sample to be written
        """
        if (self.tarstream is None or self.count >= self.maxcount
                or self.size >= self.maxsize):
            self.next_stream()
        size = self.tarstream.write(obj)
        self.count += 1
        self.total += 1
        self.size += size

    def finish(self):
        """Finish all writing (use close instead)."""
        if self.tarstream is not None:
            self.tarstream.close()
            assert self.fname is not None
            if callable(self.post):
                self.post(self.fname)
            self.tarstream = None

    def close(self):
        """Close the stream."""
        self.finish()
        del self.tarstream
        del self.shard
        del self.count
        del self.size

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, *args, **kw):
        """Exit context."""
        self.close()
