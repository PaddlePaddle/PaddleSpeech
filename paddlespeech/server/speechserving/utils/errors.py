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
import json
from enum import IntEnum

from fastapi import Response


class ErrorCode(IntEnum):
    SERVER_OK = 200  # success.

    SERVER_PARAM_ERR = 400  # Input parameters are not valid.
    SERVER_TASK_NOT_EXIST = 404  # Task is not exist.

    SERVER_INTERNAL_ERR = 500  # Internal error.
    SERVER_NETWORK_ERR = 502  # Network exception.
    SERVER_UNKOWN_ERR = 509  # Unknown error occurred.


ErrorMsg = {
    ErrorCode.SERVER_OK: "success.",
    ErrorCode.SERVER_PARAM_ERR: "Input parameters are not valid.",
    ErrorCode.SERVER_TASK_NOT_EXIST: "Task is not exist.",
    ErrorCode.SERVER_INTERNAL_ERR: "Internal error.",
    ErrorCode.SERVER_NETWORK_ERR: "Network exception.",
    ErrorCode.SERVER_UNKOWN_ERR: "Unknown error occurred."
}


def failed_response(code, msg=""):
    """Interface call failure response

    Args:
        code (int): error code number
        msg (str, optional): Interface call failure information. Defaults to "".

    Returns:
        Response (json): failure json information.
    """

    if not msg:
        msg = ErrorMsg.get(code, "Unknown error occurred.")

    res = {"success": False, "code": int(code), "message": {"description": msg}}

    return Response(content=json.dumps(res), media_type="application/json")
