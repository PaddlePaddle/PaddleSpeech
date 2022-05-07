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

from fastapi import APIRouter
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from starlette.websockets import WebSocketState as WebSocketState

from paddlespeech.cli.log import logger
from paddlespeech.server.engine.engine_pool import get_engine_pool

router = APIRouter()


@router.websocket('/ws/tts')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        # careful here, changed the source code from starlette.websockets
        assert websocket.application_state == WebSocketState.CONNECTED
        message = await websocket.receive()
        websocket._raise_on_disconnect(message)

        # get engine
        engine_pool = get_engine_pool()
        tts_engine = engine_pool['tts']

        # 获取 message 并转文本
        message = json.loads(message["text"])
        text_bese64 = message["text"]
        sentence = tts_engine.preprocess(text_bese64=text_bese64)

        # run
        wav_generator = tts_engine.run(sentence)

        while True:
            try:
                tts_results = next(wav_generator)
                resp = {"status": 1, "audio": tts_results}
                await websocket.send_json(resp)
            except StopIteration as e:
                resp = {"status": 2, "audio": ''}
                await websocket.send_json(resp)
                logger.info("Complete the transmission of audio streams")
                break

    except WebSocketDisconnect:
        pass
