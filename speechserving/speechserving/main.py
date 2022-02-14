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
import yaml
from engine.engine_factory import EngineFactory
from fastapi import FastAPI
from restful.api import setup_router

from utils.config import get_config
from utils.log import logger

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
    api_list = list(config.engine_backend)
    api_router = setup_router(api_list)
    app.include_router(api_router)

    # init engine
    engine_pool = []
    for engine in config.engine_backend:
        engine_pool.append(EngineFactory.get_engine(engine_name=engine))
        if not engine_pool[-1].init(config_file=config.engine_backend[engine]):
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
