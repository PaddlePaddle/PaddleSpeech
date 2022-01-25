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
from engine.asr.python.asr_engine import ASREngine
from fastapi import FastAPI
from restful.api import router as api_router

from utils.log import logger

app = FastAPI(
    title="PaddleSpeech Serving API", description="Api", version="0.0.1")


def init(args):
    """ 系统初始化
    """
    app.include_router(api_router)

    # engine single 
    ASR_ENGINE = ASREngine("asr")

    # todo others 

    return True


def main(args):
    """主程序入口"""

    #TODO configuration 
    from yacs.config import CfgNode
    with open(args.config_file, 'rt') as f:
        config = CfgNode(yaml.safe_load(f))

    if init(args):
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
