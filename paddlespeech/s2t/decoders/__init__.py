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
from paddlespeech.s2t.utils.log import Log

try:
    from .ctcdecoder import swig_wrapper
except:
    logger = Log(__name__).getlog()
    try:
        from paddlespeech.s2t.utils import dynamic_pip_install
        package_name = 'paddlespeech_ctcdecoders'
        dynamic_pip_install.install(package_name)
    except Exception as e:
        logger.info("paddlespeech_ctcdecoders not installed!")
