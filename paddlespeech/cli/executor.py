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
from typing import Any
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

        Args:
            tag (str): A tag of pretrained model.

        Returns:
            os.PathLike: The path on which resources of pretrained model locate. 
        """
        pass

    @abstractmethod
    def _init_from_path(self, *args, **kwargs):
        """
        Init model and other resources from arguments. This method should be called by `__call__()`.
        """
        pass

    @abstractmethod
    def preprocess(self, input: Any, *args, **kwargs):
        """
        Input preprocess and return paddle.Tensor stored in self._inputs.
        Input content can be a text(tts), a file(asr, cls), a stream(not supported yet) or anything needed.

        Args:
            input (Any): Input text/file/stream or other content.
        """
        pass

    @paddle.no_grad()
    @abstractmethod
    def infer(self, *args, **kwargs):
        """
        Model inference and put results into self._outputs.
        This method get input tensors from self._inputs, and write output tensors into self._outputs.
        """
        pass

    @abstractmethod
    def postprocess(self, *args, **kwargs) -> Union[str, os.PathLike]:
        """
        Output postprocess and return results.
        This method get model output from self._outputs and convert it into human-readable results.

        Returns:
            Union[str, os.PathLike]: Human-readable results such as texts and audio files.
        """
        pass

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
