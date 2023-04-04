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
import argparse
from abc import ABC
from abc import abstractmethod
from typing import List


class BaseExecutor(ABC):
    """
        An abstract executor of paddlespeech server tasks.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    @abstractmethod
    def execute(self, argv: List[str]) -> bool:
        """
        Command line entry. This method can only be accessed by a command line such as `paddlespeech asr`.

        Args:
            argv (List[str]): Arguments from command line.

        Returns:
            int: Result of the command execution. `True` for a success and `False` for a failure.
        """
        pass

    @abstractmethod
    def __call__(self, *arg, **kwargs):
        """
        Python API to call an executor.
        """
        pass
