# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import getpass
import logging
import os
import socket
import sys

FORMAT_STR = '[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s'
DATE_FMT_STR = '%Y/%m/%d %H:%M:%S'

logging.basicConfig(
    level=logging.DEBUG, format=FORMAT_STR, datefmt=DATE_FMT_STR)


def find_log_dir(log_dir=None):
    """Returns the most suitable directory to put log files into.
    Args:
        log_dir: str|None, if specified, the logfile(s) will be created in that
            directory.  Otherwise if the --log_dir command-line flag is provided,
            the logfile will be created in that directory.  Otherwise the logfile
            will be created in a standard location.
    Raises:
        FileNotFoundError: raised when it cannot find a log directory.
  """
    # Get a list of possible log dirs (will try to use them in order).
    if log_dir:
        # log_dir was explicitly specified as an arg, so use it and it alone.
        dirs = [log_dir]
    else:
        dirs = ['/tmp/', './']

    # Find the first usable log dir.
    for d in dirs:
        if os.path.isdir(d) and os.access(d, os.W_OK):
            return d
    raise FileNotFoundError(
        "Can't find a writable directory for logs, tried %s" % dirs)


def find_log_dir_and_names(program_name=None, log_dir=None):
    """Computes the directory and filename prefix for log file.
    Args:
        program_name: str|None, the filename part of the path to the program that
            is running without its extension.  e.g: if your program is called
            'usr/bin/foobar.py' this method should probably be called with
            program_name='foobar' However, this is just a convention, you can
            pass in any string you want, and it will be used as part of the
            log filename. If you don't pass in anything, the default behavior
            is as described in the example.  In python standard logging mode,
            the program_name will be prepended with py_ if it is the program_name
            argument is omitted.
        log_dir: str|None, the desired log directory.
    Returns:
        (log_dir, file_prefix, symlink_prefix)
    Raises:
        FileNotFoundError: raised in Python 3 when it cannot find a log directory.
        OSError: raised in Python 2 when it cannot find a log directory.
  """
    if not program_name:
        # Strip the extension (foobar.par becomes foobar, and
        # fubar.py becomes fubar). We do this so that the log
        # file names are similar to C++ log file names.
        program_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

        # Prepend py_ to files so that python code gets a unique file, and
        # so that C++ libraries do not try to write to the same log files as us.
        program_name = 'py_%s' % program_name

    actual_log_dir = find_log_dir(log_dir=log_dir)

    try:
        username = getpass.getuser()
    except KeyError:
        # This can happen, e.g. when running under docker w/o passwd file.
        if hasattr(os, 'getuid'):
            # Windows doesn't have os.getuid
            username = str(os.getuid())
        else:
            username = 'unknown'
    hostname = socket.gethostname()
    file_prefix = '%s.%s.%s.log' % (program_name, hostname, username)

    return actual_log_dir, file_prefix, program_name


class Log():

    log_name = None

    def __init__(self, logger=None):
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)

        file_dir = os.getcwd() + '/log'
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        self.log_dir = file_dir

        actual_log_dir, file_prefix, symlink_prefix = find_log_dir_and_names(
            program_name=None, log_dir=self.log_dir)

        basename = '%s.DEBUG.%d' % (file_prefix, os.getpid())
        filename = os.path.join(actual_log_dir, basename)
        if Log.log_name is None:
            Log.log_name = filename

        # Create a symlink to the log file with a canonical name.
        symlink = os.path.join(actual_log_dir, symlink_prefix + '.DEBUG')
        try:
            if os.path.islink(symlink):
                os.unlink(symlink)
            os.symlink(os.path.basename(Log.log_name), symlink)
        except EnvironmentError:
            # If it fails, we're sad but it's no error.  Commonly, this
            # fails because the symlink was created by another user and so
            # we can't modify it
            pass

        if not self.logger.hasHandlers():
            formatter = logging.Formatter(fmt=FORMAT_STR, datefmt=DATE_FMT_STR)
            fh = logging.FileHandler(Log.log_name)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        # stop propagate for propagating may print
        # log multiple times
        self.logger.propagate = False

    def getlog(self):
        return self.logger
