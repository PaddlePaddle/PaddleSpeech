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
from collections import defaultdict

__all__ = ['server_commands', 'client_commands']


def _CommandDict():
    return defaultdict(_CommandDict)


def server_execute():
    com = server_commands
    idx = 0
    for _argv in (['paddlespeech_server'] + sys.argv[1:]):
        if _argv not in com:
            break
        idx += 1
        com = com[_argv]

    # The method 'execute' of a command instance returns 'True' for a success
    # while 'False' for a failure. Here converts this result into a exit status
    # in bash: 0 for a success and 1 for a failure.
    status = 0 if com['_entry']().execute(sys.argv[idx:]) else 1
    return status


def client_execute():
    com = client_commands
    idx = 0
    for _argv in (['paddlespeech_client'] + sys.argv[1:]):
        if _argv not in com:
            break
        idx += 1
        com = com[_argv]

    # The method 'execute' of a command instance returns 'True' for a success
    # while 'False' for a failure. Here converts this result into a exit status
    # in bash: 0 for a success and 1 for a failure.
    status = 0 if com['_entry']().execute(sys.argv[idx:]) else 1
    return status


server_commands = _CommandDict()
client_commands = _CommandDict()
