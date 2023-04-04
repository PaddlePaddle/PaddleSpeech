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
import datetime
import time

from paddlespeech.s2t.utils.log import Log

__all__ = ["Timer"]

logger = Log(__name__).getlog()


class Timer():
    """To be used like this: 
        with Timer("Message") as value:
            do some thing
    """
    def __init__(self, message=None):
        self.message = message

    def duration(self) -> str:
        elapsed_time = time.time() - self.start
        time_str = str(datetime.timedelta(seconds=elapsed_time))
        return time_str

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        if self.message:
            logger.info(self.message.format(self.duration()))

    def __call__(self) -> float:
        return time.time() - self.start

    def __str__(self):
        return self.duration()
