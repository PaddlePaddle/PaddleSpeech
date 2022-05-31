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

from ..resource import CommonTaskResource
from .entry import commands
from .utils import cli_register
from .utils import explicit_command_register
from .utils import get_command

__all__ = ['BaseCommand', 'HelpCommand', 'StatsCommand']


@cli_register(name='paddlespeech')
class BaseCommand:
    def execute(self, argv: List[str]) -> bool:
        help = get_command('paddlespeech.help')
        return help().execute(argv)


@cli_register(name='paddlespeech.help', description='Show help for commands.')
class HelpCommand:
    def execute(self, argv: List[str]) -> bool:
        msg = 'Usage:\n'
        msg += '    paddlespeech <command> <options>\n\n'
        msg += 'Commands:\n'
        for command, detail in commands['paddlespeech'].items():
            if command.startswith('_'):
                continue

            if '_description' not in detail:
                continue
            msg += '    {:<15}        {}\n'.format(command,
                                                   detail['_description'])

        print(msg)
        return True


@cli_register(
    name='paddlespeech.version',
    description='Show version and commit id of current package.')
class VersionCommand:
    def execute(self, argv: List[str]) -> bool:
        try:
            from .. import __version__
            version = __version__
        except ImportError:
            version = 'Not an official release'

        try:
            from .. import __commit__
            commit_id = __commit__
        except ImportError:
            commit_id = 'Not found'

        msg = 'Package Version:\n'
        msg += '    {}\n\n'.format(version)
        msg += 'Commit ID:\n'
        msg += '    {}\n\n'.format(commit_id)

        print(msg)
        return True


model_name_format = {
    'asr': 'Model-Language-Sample Rate',
    'cls': 'Model-Sample Rate',
    'st': 'Model-Source language-Target language',
    'text': 'Model-Task-Language',
    'tts': 'Model-Language',
    'vector': 'Model-Sample Rate'
}


@cli_register(
    name='paddlespeech.stats',
    description='Get speech tasks support models list.')
class StatsCommand:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            prog='paddlespeech.stats', add_help=True)
        self.task_choices = ['asr', 'cls', 'st', 'text', 'tts', 'vector']
        self.parser.add_argument(
            '--task',
            type=str,
            default='asr',
            choices=self.task_choices,
            help='Choose speech task.',
            required=True)

    def show_support_models(self, pretrained_models: dict):
        fields = model_name_format[self.task].split("-")
        table = PrettyTable(fields)
        for key in pretrained_models:
            table.add_row(key.split("-"))
        print(table)

    def execute(self, argv: List[str]) -> bool:
        parser_args = self.parser.parse_args(argv)
        self.task = parser_args.task
        if self.task not in self.task_choices:
            print("Please input correct speech task, choices = " + str(
                self.task_choices))
            return

        pretrained_models = CommonTaskResource(task=self.task).pretrained_models

        try:
            print(
                "Here is the list of {} pretrained models released by PaddleSpeech that can be used by command line and python API"
                .format(self.task.upper()))
            self.show_support_models(pretrained_models)
        except BaseException:
            print("Failed to get the list of {} pretrained models.".format(
                self.task.upper()))


# Dynamic import when running specific command
_commands = {
    'asr': ['Speech to text infer command.', 'ASRExecutor'],
    'cls': ['Audio classification infer command.', 'CLSExecutor'],
    'st': ['Speech translation infer command.', 'STExecutor'],
    'text': ['Text command.', 'TextExecutor'],
    'tts': ['Text to Speech infer command.', 'TTSExecutor'],
    'vector': ['Speech to vector embedding infer command.', 'VectorExecutor'],
}

for com, info in _commands.items():
    explicit_command_register(
        name='paddlespeech.{}'.format(com),
        description=info[0],
        cls='paddlespeech.cli.{}.{}'.format(com, info[1]))
        