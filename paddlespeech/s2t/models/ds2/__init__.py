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
import sys

from .deepspeech2 import DeepSpeech2InferModel
from .deepspeech2 import DeepSpeech2Model
from paddlespeech.s2t.utils import dynamic_pip_install

try:
    import paddlespeech_ctcdecoders
except ImportError:
    try:
        package_name = 'paddlespeech_ctcdecoders'
        if sys.platform != "win32":
            dynamic_pip_install.install(package_name)
    except Exception:
        raise RuntimeError(
            "Can not install package paddlespeech_ctcdecoders on your system. \
                The DeepSpeech2 model is not supported for your system")

__all__ = ['DeepSpeech2Model', 'DeepSpeech2InferModel']
