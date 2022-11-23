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
import _locale

from .base_commands import ClientBaseCommand
from .base_commands import ClientHelpCommand
from .base_commands import ServerBaseCommand
from .base_commands import ServerHelpCommand
from .bin.paddlespeech_client import ASRClientExecutor
from .bin.paddlespeech_client import CLSClientExecutor
from .bin.paddlespeech_client import TTSClientExecutor
from .bin.paddlespeech_server import ServerExecutor

_locale._getdefaultlocale = (lambda *args: ['en_US', 'utf8'])
