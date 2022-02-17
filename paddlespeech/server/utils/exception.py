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
import traceback

from paddlespeech.server.utils.errors import ErrorMsg


class ServerBaseException(Exception):
    """ Server Base exception
    """

    def __init__(self, error_code, msg=None):
        #if msg:
        #log.error(msg)
        msg = msg if msg else ErrorMsg.get(error_code, "")
        super(ServerBaseException, self).__init__(error_code, msg)
        self.error_code = error_code
        self.msg = msg
        traceback.print_exc()
