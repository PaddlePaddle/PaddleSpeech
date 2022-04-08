# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import datetime
import logging
import os
import re
import sys

from config import LOGS_NUM


class MultiprocessHandler(logging.FileHandler):
    """
    A handler class which writes formatted logging records to disk files
    """

    def __init__(self,
                 filename,
                 when='D',
                 backupCount=0,
                 encoding=None,
                 delay=False):
        """
        Open the specified file and use it as the stream for logging
        """
        self.prefix = filename
        self.backupCount = backupCount
        self.when = when.upper()
        self.extMath = r"^\d{4}-\d{2}-\d{2}"

        self.when_dict = {
            'S': "%Y-%m-%d-%H-%M-%S",
            'M': "%Y-%m-%d-%H-%M",
            'H': "%Y-%m-%d-%H",
            'D': "%Y-%m-%d"
        }

        self.suffix = self.when_dict.get(when)
        if not self.suffix:
            print('The specified date interval unit is invalid: ', self.when)
            sys.exit(1)

        self.filefmt = os.path.join('.', "logs",
                                    f"{self.prefix}-{self.suffix}.log")

        self.filePath = datetime.datetime.now().strftime(self.filefmt)

        _dir = os.path.dirname(self.filefmt)
        try:
            if not os.path.exists(_dir):
                os.makedirs(_dir)
        except Exception as e:
            print('Failed to create log file: ', e)
            print("log_path：" + self.filePath)
            sys.exit(1)

        logging.FileHandler.__init__(self, self.filePath, 'a+', encoding, delay)

    def should_change_file_to_write(self):
        """
        To write the file
        """
        _filePath = datetime.datetime.now().strftime(self.filefmt)
        if _filePath != self.filePath:
            self.filePath = _filePath
            return True
        return False

    def do_change_file(self):
        """
        To change file states
        """
        self.baseFilename = os.path.abspath(self.filePath)
        if self.stream:
            self.stream.close()
            self.stream = None

        if not self.delay:
            self.stream = self._open()
        if self.backupCount > 0:
            for s in self.get_files_to_delete():
                os.remove(s)

    def get_files_to_delete(self):
        """
        To delete backup files
        """
        dir_name, _ = os.path.split(self.baseFilename)
        file_names = os.listdir(dir_name)
        result = []
        prefix = self.prefix + '-'
        for file_name in file_names:
            if file_name[:len(prefix)] == prefix:
                suffix = file_name[len(prefix):-4]
                if re.compile(self.extMath).match(suffix):
                    result.append(os.path.join(dir_name, file_name))
        result.sort()

        if len(result) < self.backupCount:
            result = []
        else:
            result = result[:len(result) - self.backupCount]
        return result

    def emit(self, record):
        """
        Emit a record
        """
        try:
            if self.should_change_file_to_write():
                self.do_change_file()
            logging.FileHandler.emit(self, record)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            self.handleError(record)


def write_log():
    """
    Init a logger
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # formatter = '%(asctime)s ｜ %(levelname)s ｜ %(filename)s ｜ %(funcName)s ｜ %(module)s ｜ %(lineno)s ｜ %(message)s'
    fmt = logging.Formatter(
        '%(asctime)s ｜ %(levelname)s ｜ %(filename)s ｜ %(funcName)s ｜ %(lineno)s ｜ %(message)s'
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)

    log_name = "audio-searching"
    file_handler = MultiprocessHandler(log_name, when='D', backupCount=LOGS_NUM)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    file_handler.do_change_file()

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


LOGGER = write_log()

if __name__ == "__main__":
    message = 'test writing logs'
    LOGGER.info(message)
    LOGGER.debug(message)
    LOGGER.error(message)
