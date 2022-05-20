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
import sys
import warnings
from typing import List

import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from prettytable import PrettyTable
from starlette.middleware.cors import CORSMiddleware

from ..executor import BaseExecutor
from ..util import cli_server_register
from ..util import stats_wrapper
from paddlespeech.cli.log import logger
from paddlespeech.server.engine.engine_pool import init_engine_pool
from paddlespeech.server.restful.api import setup_router as setup_http_router
from paddlespeech.server.utils.config import get_config
from paddlespeech.server.ws.api import setup_router as setup_ws_router
warnings.filterwarnings("ignore")

__all__ = ['ServerExecutor', 'ServerStatsExecutor']

app = FastAPI(
    title="PaddleSpeech Serving API", description="Api", version="0.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])
<<<<<<< HEAD
=======

>>>>>>> develop

@cli_server_register(
    name='paddlespeech_server.start', description='Start the service')
class ServerExecutor(BaseExecutor):
    def __init__(self):
        super(ServerExecutor, self).__init__()
        self.parser = argparse.ArgumentParser(
            prog='paddlespeech_server.start', add_help=True)
        self.parser.add_argument(
            "--config_file",
            action="store",
            help="yaml file of the app",
            default=None,
            required=True)

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
        api_list = list(engine.split("_")[0] for engine in config.engine_list)
        if config.protocol == "websocket":
            api_router = setup_ws_router(api_list)
        elif config.protocol == "http":
            api_router = setup_http_router(api_list)
        else:
            raise Exception("unsupported protocol")
        app.include_router(api_router)
        logger.info("start to init the engine")
        if not init_engine_pool(config):
            return False

        return True

    def execute(self, argv: List[str]) -> bool:
        args = self.parser.parse_args(argv)
        try:
            self(args.config_file, args.log_file)
        except Exception as e:
            logger.error("Failed to start server.")
            logger.error(e)
            sys.exit(-1)

    @stats_wrapper
    def __call__(self,
                 config_file: str="./conf/application.yaml",
                 log_file: str="./log/paddlespeech.log"):
        """
        Python API to call an executor.
        """
        config = get_config(config_file)
        if self.init(config):
            uvicorn.run(app, host=config.host, port=config.port, debug=True)


@cli_server_register(
    name='paddlespeech_server.stats',
    description='Get the models supported by each speech task in the service.')
class ServerStatsExecutor():
    def __init__(self):
        super(ServerStatsExecutor, self).__init__()

        self.parser = argparse.ArgumentParser(
            prog='paddlespeech_server.stats', add_help=True)
        self.parser.add_argument(
            '--task',
            type=str,
            default=None,
            choices=['asr', 'tts', 'cls', 'text', 'vector'],
            help='Choose speech task.',
            required=True)
        self.task_choices = ['asr', 'tts', 'cls', 'text', 'vector']
        self.model_name_format = {
            'asr': 'Model-Language-Sample Rate',
            'tts': 'Model-Language',
            'cls': 'Model-Sample Rate',
            'text': 'Model-Task-Language',
            'vector': 'Model-Sample Rate'
        }

    def show_support_models(self, pretrained_models: dict):
        fields = self.model_name_format[self.task].split("-")
        table = PrettyTable(fields)
        for key in pretrained_models:
            table.add_row(key.split("-"))
        print(table)

    def execute(self, argv: List[str]) -> bool:
        """
            Command line entry.
        """
        parser_args = self.parser.parse_args(argv)
        self.task = parser_args.task
        if self.task not in self.task_choices:
            logger.error(
                "Please input correct speech task, choices = ['asr', 'tts']")
            return False

        elif self.task.lower() == 'asr':
            try:
                from paddlespeech.cli.asr.infer import pretrained_models
                logger.info(
                    "Here is the table of ASR pretrained models supported in the service."
                )
                self.show_support_models(pretrained_models)

                # show ASR static pretrained model
                from paddlespeech.server.engine.asr.paddleinference.asr_engine import pretrained_models
                logger.info(
                    "Here is the table of ASR static pretrained models supported in the service."
                )
                self.show_support_models(pretrained_models)

                return True
            except BaseException:
                logger.error(
                    "Failed to get the table of ASR pretrained models supported in the service."
                )
                return False

        elif self.task.lower() == 'tts':
            try:
                from paddlespeech.cli.tts.infer import pretrained_models
                logger.info(
                    "Here is the table of TTS pretrained models supported in the service."
                )
                self.show_support_models(pretrained_models)

                # show TTS static pretrained model
                from paddlespeech.server.engine.tts.paddleinference.tts_engine import pretrained_models
                logger.info(
                    "Here is the table of TTS static pretrained models supported in the service."
                )
                self.show_support_models(pretrained_models)

                return True
            except BaseException:
                logger.error(
                    "Failed to get the table of TTS pretrained models supported in the service."
                )
                return False

        elif self.task.lower() == 'cls':
            try:
                from paddlespeech.cli.cls.infer import pretrained_models
                logger.info(
                    "Here is the table of CLS pretrained models supported in the service."
                )
                self.show_support_models(pretrained_models)

                # show CLS static pretrained model
                from paddlespeech.server.engine.cls.paddleinference.cls_engine import pretrained_models
                logger.info(
                    "Here is the table of CLS static pretrained models supported in the service."
                )
                self.show_support_models(pretrained_models)

                return True
            except BaseException:
                logger.error(
                    "Failed to get the table of CLS pretrained models supported in the service."
                )
                return False
        elif self.task.lower() == 'text':
            try:
                from paddlespeech.cli.text.infer import pretrained_models
                logger.info(
                    "Here is the table of Text pretrained models supported in the service."
                )
                self.show_support_models(pretrained_models)

                return True
            except BaseException:
                logger.error(
                    "Failed to get the table of Text pretrained models supported in the service."
                )
                return False
        elif self.task.lower() == 'vector':
            try:
                from paddlespeech.cli.vector.infer import pretrained_models
                logger.info(
                    "Here is the table of Vector pretrained models supported in the service."
                )
                self.show_support_models(pretrained_models)

                return True
            except BaseException:
                logger.error(
                    "Failed to get the table of Vector pretrained models supported in the service."
                )
                return False
        else:
            logger.error(
                f"Failed to get the table of {self.task} pretrained models supported in the service."
            )
            return False
