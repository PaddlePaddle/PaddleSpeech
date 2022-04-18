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

import numpy as np
from fastapi import APIRouter
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from starlette.websockets import WebSocketState as WebSocketState

from paddlespeech.server.engine.engine_pool import get_engine_pool
from paddlespeech.server.utils.buffer import ChunkBuffer
from paddlespeech.server.utils.vad import VADAudio
from paddlespeech.server.engine.asr.online.asr_engine import PaddleASRConnectionHanddler

router = APIRouter()


@router.websocket('/ws/asr')
async def websocket_endpoint(websocket: WebSocket):
    print("websocket protocal receive the dataset")
    await websocket.accept()

    engine_pool = get_engine_pool()
    asr_engine = engine_pool['asr']
    connection_handler = None
    # init buffer
    # each websocekt connection has its own chunk buffer
    chunk_buffer_conf = asr_engine.config.chunk_buffer_conf
    chunk_buffer = ChunkBuffer(
        window_n=chunk_buffer_conf.window_n,
        shift_n=chunk_buffer_conf.shift_n,
        window_ms=chunk_buffer_conf.window_ms,
        shift_ms=chunk_buffer_conf.shift_ms,
        sample_rate=chunk_buffer_conf.sample_rate,
        sample_width=chunk_buffer_conf.sample_width)

    # init vad
    vad_conf = asr_engine.config.get('vad_conf', None)
    if vad_conf:
        vad = VADAudio(
            aggressiveness=vad_conf['aggressiveness'],
            rate=vad_conf['sample_rate'],
            frame_duration_ms=vad_conf['frame_duration_ms'])

    try:
        while True:
            # careful here, changed the source code from starlette.websockets
            assert websocket.application_state == WebSocketState.CONNECTED
            message = await websocket.receive()
            websocket._raise_on_disconnect(message)
            if "text" in message:
                message = json.loads(message["text"])
                if 'signal' not in message:
                    resp = {"status": "ok", "message": "no valid json data"}
                    await websocket.send_json(resp)

                if message['signal'] == 'start':
                    resp = {"status": "ok", "signal": "server_ready"}
                    # do something at begining here
                    # create the instance to process the audio
                    connection_handler = PaddleASRConnectionHanddler(asr_engine)
                    await websocket.send_json(resp)
                elif message['signal'] == 'end':
                    engine_pool = get_engine_pool()
                    asr_engine = engine_pool['asr']
                    # reset single  engine for an new connection
                    asr_results = connection_handler.decode(is_finished=True)
                    connection_handler.reset()
                    asr_engine.reset()
                    resp = {"status": "ok", "signal": "finished", 'asr_results': asr_results}
                    await websocket.send_json(resp)
                    break
                else:
                    resp = {"status": "ok", "message": "no valid json data"}
                    await websocket.send_json(resp)
            elif "bytes" in message:
                message = message["bytes"]
                engine_pool = get_engine_pool()
                asr_engine = engine_pool['asr']
                asr_results = ""
                connection_handler.extract_feat(message)
                asr_results = connection_handler.decode(is_finished=False)
                # connection_handler.
                # frames = chunk_buffer.frame_generator(message)
                # for frame in frames:
                #     # get the pcm data from the bytes
                #     samples = np.frombuffer(frame.bytes, dtype=np.int16)
                #     sample_rate = asr_engine.config.sample_rate
                #     x_chunk, x_chunk_lens = asr_engine.preprocess(samples,
                #                                                   sample_rate)
                #     asr_engine.run(x_chunk, x_chunk_lens)
                #     asr_results = asr_engine.postprocess()

                #     # connection accept the sample data frame by frame
                    
                # asr_results = asr_engine.postprocess()
                resp = {'asr_results': asr_results}
                print("\n")
                await websocket.send_json(resp)
    except WebSocketDisconnect:
        pass
