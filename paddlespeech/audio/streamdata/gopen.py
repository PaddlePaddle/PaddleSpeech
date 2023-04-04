#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#
"""Open URLs by calling subcommands."""
import os
import re
import sys
from subprocess import PIPE
from subprocess import Popen
from urllib.parse import urlparse

# global used for printing additional node information during verbose output
info = {}


class Pipe:
    """Wrapper class for subprocess.Pipe.

    This class looks like a stream from the outside, but it checks
    subprocess status and handles timeouts with exceptions.
    This way, clients of the class do not need to know that they are
    dealing with subprocesses.

    :param *args: passed to `subprocess.Pipe`
    :param **kw: passed to `subprocess.Pipe`
    :param timeout: timeout for closing/waiting
    :param ignore_errors: don't raise exceptions on subprocess errors
    :param ignore_status: list of status codes to ignore
    """
    def __init__(
        self,
        *args,
        mode=None,
        timeout=7200.0,
        ignore_errors=False,
        ignore_status=[],
        **kw,
    ):
        """Create an IO Pipe."""
        self.ignore_errors = ignore_errors
        self.ignore_status = [0] + ignore_status
        self.timeout = timeout
        self.args = (args, kw)
        if mode[0] == "r":
            self.proc = Popen(*args, stdout=PIPE, **kw)
            self.stream = self.proc.stdout
            if self.stream is None:
                raise ValueError(f"{args}: couldn't open")
        elif mode[0] == "w":
            self.proc = Popen(*args, stdin=PIPE, **kw)
            self.stream = self.proc.stdin
            if self.stream is None:
                raise ValueError(f"{args}: couldn't open")
        self.status = None

    def __str__(self):
        return f"<Pipe {self.args}>"

    def check_status(self):
        """Poll the process and handle any errors."""
        status = self.proc.poll()
        if status is not None:
            self.wait_for_child()

    def wait_for_child(self):
        """Check the status variable and raise an exception if necessary."""
        verbose = int(os.environ.get("GOPEN_VERBOSE", 0))
        if self.status is not None and verbose:
            # print(f"(waiting again [{self.status} {os.getpid()}:{self.proc.pid}])", file=sys.stderr)
            return
        self.status = self.proc.wait()
        if verbose:
            print(
                f"pipe exit [{self.status} {os.getpid()}:{self.proc.pid}] {self.args} {info}",
                file=sys.stderr,
            )
        if self.status not in self.ignore_status and not self.ignore_errors:
            raise Exception(f"{self.args}: exit {self.status} (read) {info}")

    def read(self, *args, **kw):
        """Wrap stream.read and checks status."""
        result = self.stream.read(*args, **kw)
        self.check_status()
        return result

    def write(self, *args, **kw):
        """Wrap stream.write and checks status."""
        result = self.stream.write(*args, **kw)
        self.check_status()
        return result

    def readLine(self, *args, **kw):
        """Wrap stream.readLine and checks status."""
        result = self.stream.readLine(*args, **kw)
        self.status = self.proc.poll()
        self.check_status()
        return result

    def close(self):
        """Wrap stream.close, wait for the subprocess, and handle errors."""
        self.stream.close()
        self.status = self.proc.wait(self.timeout)
        self.wait_for_child()

    def __enter__(self):
        """Context handler."""
        return self

    def __exit__(self, etype, value, traceback):
        """Context handler."""
        self.close()


def set_options(obj,
                timeout=None,
                ignore_errors=None,
                ignore_status=None,
                handler=None):
    """Set options for Pipes.

    This function can be called on any stream. It will set pipe options only
    when its argument is a pipe.

    :param obj: any kind of stream
    :param timeout: desired timeout
    :param ignore_errors: desired ignore_errors setting
    :param ignore_status: desired ignore_status setting
    :param handler: desired error handler
    """
    if not isinstance(obj, Pipe):
        return False
    if timeout is not None:
        obj.timeout = timeout
    if ignore_errors is not None:
        obj.ignore_errors = ignore_errors
    if ignore_status is not None:
        obj.ignore_status = ignore_status
    if handler is not None:
        obj.handler = handler
    return True


def gopen_file(url, mode="rb", bufsize=8192):
    """Open a file.

    This works for local files, files over HTTP, and pipe: files.

    :param url: URL to be opened
    :param mode: mode to open it with
    :param bufsize: requested buffer size
    """
    return open(url, mode)


def gopen_pipe(url, mode="rb", bufsize=8192):
    """Use gopen to open a pipe.

    :param url: a pipe: URL
    :param mode: desired mode
    :param bufsize: desired buffer size
    """
    assert url.startswith("pipe:")
    cmd = url[5:]
    if mode[0] == "r":
        return Pipe(
            cmd,
            mode=mode,
            shell=True,
            bufsize=bufsize,
            ignore_status=[141],
        )  # skipcq: BAN-B604
    elif mode[0] == "w":
        return Pipe(
            cmd,
            mode=mode,
            shell=True,
            bufsize=bufsize,
            ignore_status=[141],
        )  # skipcq: BAN-B604
    else:
        raise ValueError(f"{mode}: unknown mode")


def gopen_curl(url, mode="rb", bufsize=8192):
    """Open a URL with `curl`.

    :param url: url (usually, http:// etc.)
    :param mode: file mode
    :param bufsize: buffer size
    """
    if mode[0] == "r":
        cmd = f"curl -s -L '{url}'"
        return Pipe(
            cmd,
            mode=mode,
            shell=True,
            bufsize=bufsize,
            ignore_status=[141, 23],
        )  # skipcq: BAN-B604
    elif mode[0] == "w":
        cmd = f"curl -s -L -T - '{url}'"
        return Pipe(
            cmd,
            mode=mode,
            shell=True,
            bufsize=bufsize,
            ignore_status=[141, 26],
        )  # skipcq: BAN-B604
    else:
        raise ValueError(f"{mode}: unknown mode")


def gopen_htgs(url, mode="rb", bufsize=8192):
    """Open a URL with `curl`.

    :param url: url (usually, http:// etc.)
    :param mode: file mode
    :param bufsize: buffer size
    """
    if mode[0] == "r":
        url = re.sub(r"(?i)^htgs://", "gs://", url)
        cmd = f"curl -s -L '{url}'"
        return Pipe(
            cmd,
            mode=mode,
            shell=True,
            bufsize=bufsize,
            ignore_status=[141, 23],
        )  # skipcq: BAN-B604
    elif mode[0] == "w":
        raise ValueError(f"{mode}: cannot write")
    else:
        raise ValueError(f"{mode}: unknown mode")


def gopen_gsutil(url, mode="rb", bufsize=8192):
    """Open a URL with `curl`.

    :param url: url (usually, http:// etc.)
    :param mode: file mode
    :param bufsize: buffer size
    """
    if mode[0] == "r":
        cmd = f"gsutil cat '{url}'"
        return Pipe(
            cmd,
            mode=mode,
            shell=True,
            bufsize=bufsize,
            ignore_status=[141, 23],
        )  # skipcq: BAN-B604
    elif mode[0] == "w":
        cmd = f"gsutil cp - '{url}'"
        return Pipe(
            cmd,
            mode=mode,
            shell=True,
            bufsize=bufsize,
            ignore_status=[141, 26],
        )  # skipcq: BAN-B604
    else:
        raise ValueError(f"{mode}: unknown mode")


def gopen_error(url, *args, **kw):
    """Raise a value error.

    :param url: url
    :param args: other arguments
    :param kw: other keywords
    """
    raise ValueError(f"{url}: no gopen handler defined")


"""A dispatch table mapping URL schemes to handlers."""
gopen_schemes = dict(
    __default__=gopen_error,
    pipe=gopen_pipe,
    http=gopen_curl,
    https=gopen_curl,
    sftp=gopen_curl,
    ftps=gopen_curl,
    scp=gopen_curl,
    gs=gopen_gsutil,
    htgs=gopen_htgs,
)


def gopen(url, mode="rb", bufsize=8192, **kw):
    """Open the URL.

    This uses the `gopen_schemes` dispatch table to dispatch based
    on scheme.

    Support for the following schemes is built-in: pipe, file,
    http, https, sftp, ftps, scp.

    When no scheme is given the url is treated as a file.

    You can use the OPEN_VERBOSE argument to get info about
    files being opened.

    :param url: the source URL
    :param mode: the mode ("rb", "r")
    :param bufsize: the buffer size
    """
    global fallback_gopen
    verbose = int(os.environ.get("GOPEN_VERBOSE", 0))
    if verbose:
        print("GOPEN", url, info, file=sys.stderr)
    assert mode in ["rb", "wb"], mode
    if url == "-":
        if mode == "rb":
            return sys.stdin.buffer
        elif mode == "wb":
            return sys.stdout.buffer
        else:
            raise ValueError(f"unknown mode {mode}")
    pr = urlparse(url)
    if pr.scheme == "":
        bufsize = int(os.environ.get("GOPEN_BUFFER", -1))
        return open(url, mode, buffering=bufsize)
    if pr.scheme == "file":
        bufsize = int(os.environ.get("GOPEN_BUFFER", -1))
        return open(pr.path, mode, buffering=bufsize)
    handler = gopen_schemes["__default__"]
    handler = gopen_schemes.get(pr.scheme, handler)
    return handler(url, mode, bufsize, **kw)


def reader(url, **kw):
    """Open url with gopen and mode "rb".

    :param url: source URL
    :param kw: other keywords forwarded to gopen
    """
    return gopen(url, "rb", **kw)
