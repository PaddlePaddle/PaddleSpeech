#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright     2019 ~ 2020    Zeng Xingui(zengxingui@baidu.com)
#
########################################################################
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

from .trainer.trainer import Trainer
from .dataset.ark_dataset import create_kaldi_ark_dataset
from .dataset.egs_dataset import create_kaldi_egs_dataset
