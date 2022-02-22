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
from typing import List

import uvicorn
from fastapi import FastAPI

from ..executor import BaseExecutor
from ..util import cli_server_register
from paddlespeech.server.engine.engine_factory import EngineFactory
from paddlespeech.server.restful.api import setup_router
from paddlespeech.server.utils.config import get_config

__all__ = ['ServerExecutor']

app = FastAPI(
    title="PaddleSpeech Serving API", description="Api", version="0.0.1")


@cli_server_register(
    name='paddlespeech_server.start', description='Start the service')
class ServerExecutor(BaseExecutor):
    def __init__(self):
        super().__init__()
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "--config_file",
            action="store",
            help="yaml file of the app",
            default="./conf/application.yaml")

        self.parser.add_argument(
            "--log_file",
            action="store",
            help="log file",
            default="./log/paddlespeech.log")

    def init(self, config) -> bool:
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
            if not engine_pool[-1].init(
                    config_file=config.engine_backend[engine]):
                return False

        return True

    def execute(self, argv: List[str]) -> bool:
        args = self.parser.parse_args(argv)
        config = get_config(args.config_file)

        if self.init(config):
            uvicorn.run(app, host=config.host, port=config.port, debug=True)v