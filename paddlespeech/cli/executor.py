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
import os
from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Union

import paddle


class BaseExecutor(ABC):
    """
        An abstract executor of paddlespeech tasks.
    """

    def __init__(self):
        self._inputs = dict()
        self._outputs = dict()

    @abstractmethod
    def _get_pretrained_path(self, tag: str) -> os.PathLike:
        """
            Download and returns pretrained resources path of current task.
        """
        pass

    @abstractmethod
    def _init_from_path(self, *args, **kwargs):
        """
            Init model and other resources from a specific path.
        """
        pass

    @abstractmethod
    def preprocess(self, input: Union[str, os.PathLike]):
        """
            Input preprocess and return paddle.Tensor stored in self.input.
            Input content can be a text(tts), a file(asr, cls) or a streaming(not supported yet).
        """
        pass

    @paddle.no_grad()
    @abstractmethod
    def infer(self, device: str):
        """
            Model inference and result stored in self.output.
        """
        pass

    @abstractmethod
    def postprocess(self) -> Union[str, os.PathLike]:
        """
            Output postprocess and return human-readable results such as texts and audio files.
        """
        pass

    @abstractmethod
    def execute(self, argv: List[str]) -> bool:
        """
            Command line entry.
        """
        pass

    @abstractmethod
    def __call__(self, *arg, **kwargs):
        """
            Python API to call an executor.
        """
        pass
