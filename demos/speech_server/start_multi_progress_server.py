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
import warnings

import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from paddlespeech.server.engine.engine_pool import init_engine_pool
from paddlespeech.server.restful.api import setup_router as setup_http_router
from paddlespeech.server.utils.config import get_config
from paddlespeech.server.ws.api import setup_router as setup_ws_router

warnings.filterwarnings("ignore")
import sys

app = FastAPI(title="PaddleSpeech Serving API",
              description="Api",
              version="0.0.1")
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])

# change yaml file here
config_file = "./conf/application.yaml"
config = get_config(config_file)

# init engine
if not init_engine_pool(config):
    print("Failed to init engine.")
    sys.exit(-1)

# get api_router
api_list = list(engine.split("_")[0] for engine in config.engine_list)
if config.protocol == "websocket":
    api_router = setup_ws_router(api_list)
elif config.protocol == "http":
    api_router = setup_http_router(api_list)
else:
    raise Exception("unsupported protocol")
    sys.exit(-1)

# app needs to operate outside the main function
app.include_router(api_router)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--workers",
                        type=int,
                        help="workers of server",
                        default=1)
    args = parser.parse_args()

    uvicorn.run("start_multi_progress_server:app",
                host=config.host,
                port=config.port,
                debug=True,
                workers=args.workers)
