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
import argparse

import uvicorn
from fastapi import FastAPI

from paddlespeech.server.engine.engine_pool import init_engine_pool
from paddlespeech.server.restful.api import setup_router as setup_http_router
from paddlespeech.server.utils.config import get_config
from paddlespeech.server.ws.api import setup_router as setup_ws_router

app = FastAPI(
    title="PaddleSpeech Serving API", description="Api", version="0.0.1")


def init(config):
    """system initialization

    Args:
        config (CfgNode): config object

    Returns:
        bool: 
    """
    # init api
    api_list = list(engine.split("_")[0] for engine in config.engine_list)
    if config.protocol == "websocket":
        api_router = setup_ws_router(api_list)
    elif config.protocol == "http":
        api_router = setup_http_router(api_list)
    else:
        raise Exception("unsupported protocol")
    app.include_router(api_router)

    if not init_engine_pool(config):
        return False

    return True


def main(args):
    """main function"""

    config = get_config(args.config_file)

    if init(config):
        uvicorn.run(app, host=config.host, port=config.port, debug=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        action="store",
        help="yaml file of the app",
        default="./conf/application.yaml")

    parser.add_argument(
        "--log_file",
        action="store",
        help="log file",
        default="./log/paddlespeech.log")
    args = parser.parse_args()

    main(args)
