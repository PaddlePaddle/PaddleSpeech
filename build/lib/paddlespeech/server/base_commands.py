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

from .entry import client_commands
from .entry import server_commands
from .util import cli_client_register
from .util import cli_server_register
from .util import get_client_command
from .util import get_server_command

__all__ = [
    'ServerBaseCommand',
    'ServerHelpCommand',
    'ClientBaseCommand',
    'ClientHelpCommand',
]


@cli_server_register(name='paddlespeech_server')
class ServerBaseCommand:
    def execute(self, argv: List[str]) -> bool:
        help = get_server_command('paddlespeech_server.help')
        return help().execute(argv)


@cli_server_register(
    name='paddlespeech_server.help', description='Show help for commands.')
class ServerHelpCommand:
    def execute(self, argv: List[str]) -> bool:
        msg = 'Usage:\n'
        msg += '    paddlespeech_server <command> <options>\n\n'
        msg += 'Commands:\n'
        for command, detail in server_commands['paddlespeech_server'].items():
            if command.startswith('_'):
                continue

            if '_description' not in detail:
                continue
            msg += '    {:<15}        {}\n'.format(command,
                                                   detail['_description'])

        print(msg)
        return True


@cli_client_register(name='paddlespeech_client')
class ClientBaseCommand:
    def execute(self, argv: List[str]) -> bool:
        help = get_client_command('paddlespeech_client.help')
        return help().execute(argv)


@cli_client_register(
    name='paddlespeech_client.help', description='Show help for commands.')
class ClientHelpCommand:
    def execute(self, argv: List[str]) -> bool:
        msg = 'Usage:\n'
        msg += '    paddlespeech_client <command> <options>\n\n'
        msg += 'Commands:\n'
        for command, detail in client_commands['paddlespeech_client'].items():
            if command.startswith('_'):
                continue

            if '_description' not in detail:
                continue
            msg += '    {:<15}        {}\n'.format(command,
                                                   detail['_description'])

        print(msg)
        return True
