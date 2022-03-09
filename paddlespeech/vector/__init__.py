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

"""
__init__ file for sidt package.
"""

import logging as sidt_logging
import colorlog

LOG_COLOR_CONFIG = {
    'DEBUG': 'white',
    'INFO': 'white',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'purple',
}

# 设置全局的logger
colored_formatter = colorlog.ColoredFormatter(
    '%(log_color)s [%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors=LOG_COLOR_CONFIG)  # 日志输出格式
_logger = sidt_logging.getLogger("sidt")
handler = colorlog.StreamHandler()
handler.setLevel(sidt_logging.INFO)
handler.setFormatter(colored_formatter)
_logger.addHandler(handler)
_logger.setLevel(sidt_logging.INFO)

