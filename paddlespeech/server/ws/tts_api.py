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
import uuid

from fastapi import APIRouter
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from starlette.websockets import WebSocketState as WebSocketState

from paddlespeech.cli.log import logger
from paddlespeech.server.engine.engine_pool import get_engine_pool

router = APIRouter()


@router.websocket('/paddlespeech/tts/streaming')
async def websocket_endpoint(websocket: WebSocket):
    """PaddleSpeech Online TTS Server api

    Args:
        websocket (WebSocket): the websocket instance
    """

    #1. the interface wait to accept the websocket protocal header
    #   and only we receive the header, it establish the connection with specific thread
    await websocket.accept()

    #2. if we accept the websocket headers, we will get the online tts engine instance
    engine_pool = get_engine_pool()
    tts_engine = engine_pool['tts']

    try:
        while True:
            # careful here, changed the source code from starlette.websockets
            assert websocket.application_state == WebSocketState.CONNECTED
            message = await websocket.receive()
            websocket._raise_on_disconnect(message)
            message = json.loads(message["text"])

            if 'signal' in message:
                # start request
                if message['signal'] == 'start':
                    session = uuid.uuid1().hex
                    resp = {
                        "status": 0,
                        "signal": "server ready",
                        "session": session
                    }
                    await websocket.send_json(resp)

                # end request
                elif message['signal'] == 'end':
                    resp = {
                        "status": 0,
                        "signal": "connection will be closed",
                        "session": session
                    }
                    await websocket.send_json(resp)
                    break
                else:
                    resp = {"status": 0, "signal": "no valid json data"}
                    await websocket.send_json(resp)

            # speech synthesis request 
            elif 'text' in message:
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
                        import pdb
                        pdb.set_trace()
                        resp = {"status": 2, "audio": ''}
                        await websocket.send_json(resp)
                        logger.info(
                            "Complete the synthesis of the audio streams")
                        break

            else:
                logger.error(
                    "Invalid request, please check if the request is correct.")

    except WebSocketDisconnect:
        pass
