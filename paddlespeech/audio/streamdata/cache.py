# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# See the LICENSE file for licensing terms (BSD-style).
# Modified from https://github.com/webdataset/webdataset
import os
import random
import re
import sys
from urllib.parse import urlparse

from . import filters
from . import gopen
from .handlers import reraise_exception
from .tariterators import tar_file_and_group_expander

default_cache_dir = os.environ.get("WDS_CACHE", "./_cache")
default_cache_size = float(os.environ.get("WDS_CACHE_SIZE", "1e18"))


def lru_cleanup(cache_dir, cache_size, keyfn=os.path.getctime, verbose=False):
    """Performs cleanup of the file cache in cache_dir using an LRU strategy,
    keeping the total size of all remaining files below cache_size."""
    if not os.path.exists(cache_dir):
        return
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(cache_dir):
        for filename in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, filename))
    if total_size <= cache_size:
        return
    # sort files by last access time
    files = []
    for dirpath, dirnames, filenames in os.walk(cache_dir):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
    files.sort(key=keyfn, reverse=True)
    # delete files until we're under the cache size
    while len(files) > 0 and total_size > cache_size:
        fname = files.pop()
        total_size -= os.path.getsize(fname)
        if verbose:
            print("# deleting %s" % fname, file=sys.stderr)
        os.remove(fname)


def download(url, dest, chunk_size=1024**2, verbose=False):
    """Download a file from `url` to `dest`."""
    temp = dest + f".temp{os.getpid()}"
    with gopen.gopen(url) as stream:
        with open(temp, "wb") as f:
            while True:
                data = stream.read(chunk_size)
                if not data:
                    break
                f.write(data)
    os.rename(temp, dest)


def pipe_cleaner(spec):
    """Guess the actual URL from a "pipe:" specification."""
    if spec.startswith("pipe:"):
        spec = spec[5:]
        words = spec.split(" ")
        for word in words:
            if re.match(r"^(https?|gs|ais|s3)", word):
                return word
    return spec


def get_file_cached(
        spec,
        cache_size=-1,
        cache_dir=None,
        url_to_name=pipe_cleaner,
        verbose=False, ):
    if cache_size == -1:
        cache_size = default_cache_size
    if cache_dir is None:
        cache_dir = default_cache_dir
    url = url_to_name(spec)
    parsed = urlparse(url)
    dirname, filename = os.path.split(parsed.path)
    dirname = dirname.lstrip("/")
    dirname = re.sub(r"[:/|;]", "_", dirname)
    destdir = os.path.join(cache_dir, dirname)
    os.makedirs(destdir, exist_ok=True)
    dest = os.path.join(cache_dir, dirname, filename)
    if not os.path.exists(dest):
        if verbose:
            print("# downloading %s to %s" % (url, dest), file=sys.stderr)
        lru_cleanup(cache_dir, cache_size, verbose=verbose)
        download(spec, dest, verbose=verbose)
    return dest


def get_filetype(fname):
    with os.popen("file '%s'" % fname) as f:
        ftype = f.read()
    return ftype


def check_tar_format(fname):
    """Check whether a file is a tar archive."""
    ftype = get_filetype(fname)
    return "tar archive" in ftype or "gzip compressed" in ftype


verbose_cache = int(os.environ.get("WDS_VERBOSE_CACHE", "0"))


def cached_url_opener(
        data,
        handler=reraise_exception,
        cache_size=-1,
        cache_dir=None,
        url_to_name=pipe_cleaner,
        validator=check_tar_format,
        verbose=False,
        always=False, ):
    """Given a stream of url names (packaged in `dict(url=url)`), yield opened streams."""
    verbose = verbose or verbose_cache
    for sample in data:
        assert isinstance(sample, dict), sample
        assert "url" in sample
        url = sample["url"]
        attempts = 5
        try:
            if not always and os.path.exists(url):
                dest = url
            else:
                dest = get_file_cached(
                    url,
                    cache_size=cache_size,
                    cache_dir=cache_dir,
                    url_to_name=url_to_name,
                    verbose=verbose, )
            if verbose:
                print("# opening %s" % dest, file=sys.stderr)
            assert os.path.exists(dest)
            if not validator(dest):
                ftype = get_filetype(dest)
                with open(dest, "rb") as f:
                    data = f.read(200)
                os.remove(dest)
                raise ValueError(
                    "%s (%s) is not a tar archive, but a %s, contains %s" %
                    (dest, url, ftype, repr(data)))
            try:
                stream = open(dest, "rb")
                sample.update(stream=stream)
                yield sample
            except FileNotFoundError as exn:
                # dealing with race conditions in lru_cleanup
                attempts -= 1
                if attempts > 0:
                    time.sleep(random.random() * 10)
                    continue
                raise exn
        except Exception as exn:
            exn.args = exn.args + (url, )
            if handler(exn):
                continue
            else:
                break


def cached_tarfile_samples(
        src,
        handler=reraise_exception,
        cache_size=-1,
        cache_dir=None,
        verbose=False,
        url_to_name=pipe_cleaner,
        always=False, ):
    streams = cached_url_opener(
        src,
        handler=handler,
        cache_size=cache_size,
        cache_dir=cache_dir,
        verbose=verbose,
        url_to_name=url_to_name,
        always=always, )
    samples = tar_file_and_group_expander(streams, handler=handler)
    return samples


cached_tarfile_to_samples = filters.pipelinefilter(cached_tarfile_samples)
