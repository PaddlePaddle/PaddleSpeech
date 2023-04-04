#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
# Modified from https://github.com/webdataset/webdataset
# Modified from wenet(https://github.com/wenet-e2e/wenet)
"""Low level iteration functions for tar archives."""
import random
import re
import tarfile

import braceexpand

from . import filters
from . import gopen
from .handlers import reraise_exception

trace = False
meta_prefix = "__"
meta_suffix = "__"

import paddleaudio
import paddle
import numpy as np

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


def base_plus_ext(path):
    """Split off all file extensions.

    Returns base, allext.

    :param path: path with extensions
    :param returns: path with all extensions removed

    """
    match = re.match(r"^((?:.*/|)[^.]+)[.]([^/]*)$", path)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def valid_sample(sample):
    """Check whether a sample is valid.

    :param sample: sample to be checked
    """
    return (sample is not None and isinstance(sample, dict)
            and len(list(sample.keys())) > 0
            and not sample.get("__bad__", False))


# FIXME: UNUSED
def shardlist(urls, *, shuffle=False):
    """Given a list of URLs, yields that list, possibly shuffled."""
    if isinstance(urls, str):
        urls = braceexpand.braceexpand(urls)
    else:
        urls = list(urls)
    if shuffle:
        random.shuffle(urls)
    for url in urls:
        yield dict(url=url)


def url_opener(data, handler=reraise_exception, **kw):
    """Given a stream of url names (packaged in `dict(url=url)`), yield opened streams."""
    for sample in data:
        assert isinstance(sample, dict), sample
        assert "url" in sample
        url = sample["url"]
        try:
            stream = gopen.gopen(url, **kw)
            sample.update(stream=stream)
            yield sample
        except Exception as exn:
            exn.args = exn.args + (url, )
            if handler(exn):
                continue
            else:
                break


def tar_file_iterator(fileobj,
                      skip_meta=r"__[^/]*__($|/)",
                      handler=reraise_exception):
    """Iterate over tar file, yielding filename, content pairs for the given tar stream.

    :param fileobj: byte stream suitable for tarfile
    :param skip_meta: regexp for keys that are skipped entirely (Default value = r"__[^/]*__($|/)")

    """
    stream = tarfile.open(fileobj=fileobj, mode="r:*")
    for tarinfo in stream:
        fname = tarinfo.name
        try:
            if not tarinfo.isreg():
                continue
            if fname is None:
                continue
            if ("/" not in fname and fname.startswith(meta_prefix)
                    and fname.endswith(meta_suffix)):
                # skipping metadata for now
                continue
            if skip_meta is not None and re.match(skip_meta, fname):
                continue

            name = tarinfo.name
            pos = name.rfind('.')
            assert pos > 0
            prefix, postfix = name[:pos], name[pos + 1:]
            if postfix == 'wav':
                waveform, sample_rate = paddleaudio.backends.soundfile_load(
                    stream.extractfile(tarinfo), normal=False)
                result = dict(fname=prefix,
                              wav=waveform,
                              sample_rate=sample_rate)
            else:
                txt = stream.extractfile(tarinfo).read().decode('utf8').strip()
                result = dict(fname=prefix, txt=txt)
            #result = dict(fname=fname, data=data)
            yield result
            stream.members = []
        except Exception as exn:
            if hasattr(exn, "args") and len(exn.args) > 0:
                exn.args = (exn.args[0] + " @ " + str(fileobj), ) + exn.args[1:]
            if handler(exn):
                continue
            else:
                break
    del stream


def tar_file_and_group_iterator(fileobj,
                                skip_meta=r"__[^/]*__($|/)",
                                handler=reraise_exception):
    """ Expand a stream of open tar files into a stream of tar file contents.
        And groups the file with same prefix

        Args:
            data: Iterable[{src, stream}]

        Returns:
            Iterable[{key, wav, txt, sample_rate}]
    """
    stream = tarfile.open(fileobj=fileobj, mode="r:*")
    prev_prefix = None
    example = {}
    valid = True
    for tarinfo in stream:
        name = tarinfo.name
        pos = name.rfind('.')
        assert pos > 0
        prefix, postfix = name[:pos], name[pos + 1:]
        if prev_prefix is not None and prefix != prev_prefix:
            example['fname'] = prev_prefix
            if valid:
                yield example
            example = {}
            valid = True
        with stream.extractfile(tarinfo) as file_obj:
            try:
                if postfix == 'txt':
                    example['txt'] = file_obj.read().decode('utf8').strip()
                elif postfix in AUDIO_FORMAT_SETS:
                    waveform, sample_rate = paddleaudio.backends.soundfile_load(
                        file_obj, normal=False)
                    waveform = paddle.to_tensor(np.expand_dims(
                        np.array(waveform), 0),
                                                dtype=paddle.float32)

                    example['wav'] = waveform
                    example['sample_rate'] = sample_rate
                else:
                    example[postfix] = file_obj.read()
            except Exception as exn:
                if hasattr(exn, "args") and len(exn.args) > 0:
                    exn.args = (exn.args[0] + " @ " +
                                str(fileobj), ) + exn.args[1:]
                if handler(exn):
                    continue
                else:
                    break
                valid = False
            #  logging.warning('error to parse {}'.format(name))
        prev_prefix = prefix
    if prev_prefix is not None:
        example['fname'] = prev_prefix
        yield example
    stream.close()


def tar_file_expander(data, handler=reraise_exception):
    """Expand a stream of open tar files into a stream of tar file contents.

    This returns an iterator over (filename, file_contents).
    """
    for source in data:
        url = source["url"]
        try:
            assert isinstance(source, dict)
            assert "stream" in source
            for sample in tar_file_iterator(source["stream"]):
                assert (isinstance(sample, dict) and "data" in sample
                        and "fname" in sample)
                sample["__url__"] = url
                yield sample
        except Exception as exn:
            exn.args = exn.args + (source.get("stream"), source.get("url"))
            if handler(exn):
                continue
            else:
                break


def tar_file_and_group_expander(data, handler=reraise_exception):
    """Expand a stream of open tar files into a stream of tar file contents.

    This returns an iterator over (filename, file_contents).
    """
    for source in data:
        url = source["url"]
        try:
            assert isinstance(source, dict)
            assert "stream" in source
            for sample in tar_file_and_group_iterator(source["stream"]):
                assert (isinstance(sample, dict) and "wav" in sample
                        and "txt" in sample and "fname" in sample)
                sample["__url__"] = url
                yield sample
        except Exception as exn:
            exn.args = exn.args + (source.get("stream"), source.get("url"))
            if handler(exn):
                continue
            else:
                break


def group_by_keys(data,
                  keys=base_plus_ext,
                  lcase=True,
                  suffixes=None,
                  handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if trace:
            print(
                prefix,
                suffix,
                current_sample.keys()
                if isinstance(current_sample, dict) else None,
            )
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        if current_sample is None or prefix != current_sample["__key__"]:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffix in current_sample:
            raise ValueError(
                f"{fname}: duplicate file name in tar file {suffix} {current_sample.keys()}"
            )
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_samples(src, handler=reraise_exception):
    streams = url_opener(src, handler=handler)
    samples = tar_file_and_group_expander(streams, handler=handler)
    return samples


tarfile_to_samples = filters.pipelinefilter(tarfile_samples)
