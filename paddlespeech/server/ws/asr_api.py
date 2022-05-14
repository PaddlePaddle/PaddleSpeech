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
import base64
from typing import Union
from fastapi import APIRouter
from fastapi import WebSocket
import soundfile
import io
from fastapi import WebSocketDisconnect
from starlette.websockets import WebSocketState as WebSocketState

from paddlespeech.cli.log import logger
from paddlespeech.server.engine.asr.online.asr_engine import PaddleASRConnectionHanddler
from paddlespeech.server.engine.engine_pool import get_engine_pool
from paddlespeech.server.restful.response import ASRResponse
from paddlespeech.server.restful.response import ErrorResponse
from paddlespeech.server.restful.request import ASRRequest
from paddlespeech.server.utils.exception import ServerBaseException
from paddlespeech.server.utils.errors import failed_response
from paddlespeech.server.utils.errors import ErrorCode
router = APIRouter()


@router.websocket('/paddlespeech/asr/streaming')
async def websocket_endpoint(websocket: WebSocket):
    """PaddleSpeech Online ASR Server api

    Args:
        websocket (WebSocket): the websocket instance
    """

    #1. the interface wait to accept the websocket protocal header
    #   and only we receive the header, it establish the connection with specific thread
    await websocket.accept()

    #2. if we accept the websocket headers, we will get the online asr engine instance
    engine_pool = get_engine_pool()
    asr_engine = engine_pool['asr']

    #3. each websocket connection, we will create an PaddleASRConnectionHanddler to process such audio
    #   and each connection has its own connection instance to process the request
    #   and only if client send the start signal, we create the PaddleASRConnectionHanddler instance
    connection_handler = None

    try:
        #4. we do a loop to process the audio package by package according the protocal
        #   and only if the client send finished signal, we will break the loop
        while True:
            # careful here, changed the source code from starlette.websockets
            # 4.1 we wait for the client signal for the specific action
            assert websocket.application_state == WebSocketState.CONNECTED
            message = await websocket.receive()
            websocket._raise_on_disconnect(message)

            #4.2 text for the action command and bytes for pcm data
            if "text" in message:
                # we first parse the specific command
                message = json.loads(message["text"])
                if 'signal' not in message:
                    resp = {"status": "ok", "message": "no valid json data"}
                    await websocket.send_json(resp)

                # start command, we create the PaddleASRConnectionHanddler instance to process the audio data
                # end command, we process the all the last audio pcm and return the final result
                #              and we break the loop
                if message['signal'] == 'start':
                    resp = {"status": "ok", "signal": "server_ready"}
                    # do something at begining here
                    # create the instance to process the audio
                    connection_handler = PaddleASRConnectionHanddler(asr_engine)
                    await websocket.send_json(resp)
                elif message['signal'] == 'end':
                    # reset single  engine for an new connection
                    # and we will destroy the connection
                    connection_handler.decode(is_finished=True)
                    connection_handler.rescoring()
                    asr_results = connection_handler.get_result()
                    word_time_stamp = connection_handler.get_word_time_stamp()
                    connection_handler.reset()

                    resp = {
                        "status": "ok",
                        "signal": "finished",
                        'result': asr_results,
                        'times': word_time_stamp
                    }
                    await websocket.send_json(resp)
                    break
                else:
                    resp = {"status": "ok", "message": "no valid json data"}
                    await websocket.send_json(resp)
            elif "bytes" in message:
                # bytes for the pcm data
                message = message["bytes"]

                # we extract the remained audio pcm 
                # and decode for the result in this package data
                connection_handler.extract_feat(message)
                connection_handler.decode(is_finished=False)
                asr_results = connection_handler.get_result()

                # return the current period result
                # if the engine create the vad instance, this connection will have many period results 
                resp = {'result': asr_results}
                await websocket.send_json(resp)
    except WebSocketDisconnect as e:
        logger.error(e)


# @router.post(
#     "/paddlespeech/asr/search/", response_model=Union[ASRResponse, ErrorResponse])
# def asr(request_body: ASRRequest):
#     """asr api 

#     Args:
#         request_body (ASRRequest): [description]

#     Returns:
#         json: [description]
#     """
#     try:
#         audio_data = base64.b64decode(request_body.audio)

#         # get single engine from engine pool
#         engine_pool = get_engine_pool()
#         asr_engine = engine_pool['asr']

#         samples, sample_rate = soundfile.read(io.BytesIO(audio_data), dtype='int16')
#         # print(samples.shape)
#         # print(sample_rate)
#         connection_handler = PaddleASRConnectionHanddler(asr_engine)
#         connection_handler.extract_feat(samples)
        
#         connection_handler.decode(is_finished=True)
#         asr_results = connection_handler.rescoring()
#         asr_results = connection_handler.get_result()
#         word_time_stamp = connection_handler.get_word_time_stamp()

#         response = {
#             "success": True,
#             "code": 200,
#             "message": {
#                 "description": "success"
#             },
#             "result": {
#                 "transcription": asr_results,
#                 "times": word_time_stamp
#             }
#         }

        
#     except ServerBaseException as e:
#         response = failed_response(e.error_code, e.msg)
#     except BaseException as e:
#         response = failed_response(ErrorCode.SERVER_UNKOWN_ERR)
#         print(e)

#     return response