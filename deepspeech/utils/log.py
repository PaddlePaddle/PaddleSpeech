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

import logging
import os


class Log():
    def __init__(self, logger=None, log_cate='global'):
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)

        file_dir = os.getcwd() + '/log'
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        self.log_path = file_dir
        self.log_name = self.log_path + "/" + log_cate + '.log'

        fh = logging.FileHandler(self.log_name)
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        format = '[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s'
        formatter = logging.Formatter(fmt=format, datefmt='%Y/%m/%d %H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # stop propagate for propagating may print
        # log multiple times
        # self.logger.propagate = False

        fh.close()
        ch.close()

    def getlog(self):
        return self.logger
