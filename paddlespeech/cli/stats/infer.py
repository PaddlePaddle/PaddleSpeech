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
from typing import List

from prettytable import PrettyTable

from ..log import logger
from ..utils import cli_register
from ..utils import stats_wrapper

__all__ = ['StatsExecutor']

model_name_format = {
    'asr': 'Model-Language-Sample Rate',
    'cls': 'Model-Sample Rate',
    'st': 'Model-Source language-Target language',
    'text': 'Model-Task-Sample Rate',
    'tts': 'Model-Language'
}


@cli_register(
    name='paddlespeech.stats',
    description='Get speech tasks support models list.')
class StatsExecutor():
    def __init__(self):
        super(StatsExecutor, self).__init__()

        self.parser = argparse.ArgumentParser(
            prog='paddlespeech.stats', add_help=True)
        self.parser.add_argument(
            '--task',
            type=str,
            default='asr',
            choices=['asr', 'cls', 'st', 'text', 'tts'],
            help='Choose speech task.',
            required=True)
        self.task_choices = ['asr', 'cls', 'st', 'text', 'tts']

    def show_support_models(self, pretrained_models: dict):
        fields = model_name_format[self.task].split("-")
        table = PrettyTable(fields)
        for key in pretrained_models:
            table.add_row(key.split("-"))
        print(table)

    def execute(self, argv: List[str]) -> bool:
        """
            Command line entry.
        """
        parser_args = self.parser.parse_args(argv)
        self.task = parser_args.task
        if self.task not in self.task_choices:
            logger.error(
                "Please input correct speech task, choices = ['asr', 'cls', 'st', 'text', 'tts']"
            )
            return False

        if self.task == 'asr':
            try:
                from ..asr.infer import pretrained_models
                logger.info(
                    "Here is the list of ASR pretrained models released by PaddleSpeech that can be used by command line and python API"
                )
                self.show_support_models(pretrained_models)
                return True
            except BaseException:
                logger.error("Failed to get the list of ASR pretrained models.")
                return False

        elif self.task == 'cls':
            try:
                from ..cls.infer import pretrained_models
                logger.info(
                    "Here is the list of CLS pretrained models released by PaddleSpeech that can be used by command line and python API"
                )
                self.show_support_models(pretrained_models)
                return True
            except BaseException:
                logger.error("Failed to get the list of CLS pretrained models.")
                return False

        elif self.task == 'st':
            try:
                from ..st.infer import pretrained_models
                logger.info(
                    "Here is the list of ST pretrained models released by PaddleSpeech that can be used by command line and python API"
                )
                self.show_support_models(pretrained_models)
                return True
            except BaseException:
                logger.error("Failed to get the list of ST pretrained models.")
                return False

        elif self.task == 'text':
            try:
                from ..text.infer import pretrained_models
                logger.info(
                    "Here is the list of TEXT pretrained models released by PaddleSpeech that can be used by command line and python API"
                )
                self.show_support_models(pretrained_models)
                return True
            except BaseException:
                logger.error(
                    "Failed to get the list of TEXT pretrained models.")
                return False

        elif self.task == 'tts':
            try:
                from ..tts.infer import pretrained_models
                logger.info(
                    "Here is the list of TTS pretrained models released by PaddleSpeech that can be used by command line and python API"
                )
                self.show_support_models(pretrained_models)
                return True
            except BaseException:
                logger.error("Failed to get the list of TTS pretrained models.")
                return False

    @stats_wrapper
    def __call__(
            self,
            task: str=None, ):
        """
            Python API to call an executor.
        """
        self.task = task
        if self.task not in self.task_choices:
            print(
                "Please input correct speech task, choices = ['asr', 'cls', 'st', 'text', 'tts']"
            )

        elif self.task == 'asr':
            try:
                from ..asr.infer import pretrained_models
                print(
                    "Here is the list of ASR pretrained models released by PaddleSpeech that can be used by command line and python API"
                )
                self.show_support_models(pretrained_models)
            except BaseException:
                print("Failed to get the list of ASR pretrained models.")

        elif self.task == 'cls':
            try:
                from ..cls.infer import pretrained_models
                print(
                    "Here is the list of CLS pretrained models released by PaddleSpeech that can be used by command line and python API"
                )
                self.show_support_models(pretrained_models)
            except BaseException:
                print("Failed to get the list of CLS pretrained models.")

        elif self.task == 'st':
            try:
                from ..st.infer import pretrained_models
                print(
                    "Here is the list of ST pretrained models released by PaddleSpeech that can be used by command line and python API"
                )
                self.show_support_models(pretrained_models)
            except BaseException:
                print("Failed to get the list of ST pretrained models.")

        elif self.task == 'text':
            try:
                from ..text.infer import pretrained_models
                print(
                    "Here is the list of TEXT pretrained models released by PaddleSpeech that can be used by command line and python API"
                )
                self.show_support_models(pretrained_models)
            except BaseException:
                print(
                    "Failed to get the list of TEXT pretrained models.")

        elif self.task == 'tts':
            try:
                from ..tts.infer import pretrained_models
                print(
                    "Here is the list of TTS pretrained models released by PaddleSpeech that can be used by command line and python API"
                )
                self.show_support_models(pretrained_models)
            except BaseException:
                print("Failed to get the list of TTS pretrained models.")