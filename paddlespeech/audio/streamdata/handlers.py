#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#

"""Pluggable exception handlers.

These are functions that take an exception as an argument and then return...

- the exception (in order to re-raise it)
- True (in order to continue and ignore the exception)
- False (in order to ignore the exception and stop processing)

They are used as handler= arguments in much of the library.
"""

import time, warnings


def reraise_exception(exn):
    """Call in an exception handler to re-raise the exception."""
    raise exn


def ignore_and_continue(exn):
    """Call in an exception handler to ignore any exception and continue."""
    return True


def warn_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    warnings.warn(repr(exn))
    time.sleep(0.5)
    return True


def ignore_and_stop(exn):
    """Call in an exception handler to ignore any exception and stop further processing."""
    return False


def warn_and_stop(exn):
    """Call in an exception handler to ignore any exception and stop further processing."""
    warnings.warn(repr(exn))
    time.sleep(0.5)
    return False
