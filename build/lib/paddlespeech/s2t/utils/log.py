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
import inspect
import os
import socket
import sys

from loguru import logger
from paddle import inference


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
    """Default Logger for all."""
    logger.remove()

    _call_from_cli = False
    _frame = inspect.currentframe()
    while _frame:
        if 'paddlespeech/cli/entry.py' in _frame.f_code.co_filename or 'paddlespeech/t2s' in _frame.f_code.co_filename:
            _call_from_cli = True
            break
        _frame = _frame.f_back

    if _call_from_cli:
        logger.add(
            sys.stdout,
            level='ERROR',
            enqueue=True,
            filter=lambda record: record['level'].no >= 20)
    else:
        logger.add(
            sys.stdout,
            level='INFO',
            enqueue=True,
            filter=lambda record: record['level'].no >= 20)
        _, file_prefix, _ = find_log_dir_and_names()
        sink_prefix = os.path.join("exp/log", file_prefix)
        sink_path = sink_prefix[:-3] + "{time}.log"
        logger.add(sink_path, level='DEBUG', enqueue=True, rotation="500 MB")

    def __init__(self, name=None):
        pass

    def getlog(self):
        return logger


class Autolog:
    """Just used by fullchain project"""

    def __init__(self,
                 batch_size,
                 model_name="DeepSpeech",
                 model_precision="fp32"):
        import auto_log
        pid = os.getpid()
        if os.environ.get('CUDA_VISIBLE_DEVICES', None):
            gpu_id = int(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0])
            infer_config = inference.Config()
            infer_config.enable_use_gpu(100, gpu_id)
        else:
            gpu_id = None
            infer_config = inference.Config()

        self.autolog = auto_log.AutoLogger(
            model_name=model_name,
            model_precision=model_precision,
            batch_size=batch_size,
            data_shape="dynamic",
            save_path="./output/auto_log.lpg",
            inference_config=infer_config,
            pids=pid,
            process_name=None,
            gpu_ids=gpu_id,
            time_keys=['preprocess_time', 'inference_time', 'postprocess_time'],
            warmup=0)

    def getlog(self):
        return self.autolog
