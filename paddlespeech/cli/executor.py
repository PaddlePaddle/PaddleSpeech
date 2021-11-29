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
from typing import Optional
from typing import Union

import paddle


class BaseExecutor(ABC):
    """
        An abstract executor of paddlespeech tasks.
    """

    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def _get_default_cfg_path(self):
        """
            Returns a default config file path of current task.
        """
        pass

    @abstractmethod
    def _init_from_cfg(self, cfg_path: Optional[os.PathLike]=None):
        """
            Init model from a specific config file.
        """
        pass

    @abstractmethod
    def preprocess(self, input: Union[str, os.PathLike]):
        """
            Input preprocess and return paddle.Tensor stored in self.input.
            Input content can be a text(t2s), a file(s2t, cls) or a streaming(not supported yet).
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
