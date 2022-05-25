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
from typing import List

from .entry import commands
from .utils import cli_register
from .utils import explicit_command_register
from .utils import get_command

__all__ = [
    'BaseCommand',
    'HelpCommand',
]


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
