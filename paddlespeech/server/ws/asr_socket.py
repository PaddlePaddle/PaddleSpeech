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

router = APIRouter()


@router.websocket('/ws/asr')
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()

    engine_pool = get_engine_pool()
    asr_engine = engine_pool['asr']
    # init buffer
    chunk_buffer_conf = asr_engine.config.chunk_buffer_conf
    chunk_buffer = ChunkBuffer(
        sample_rate=chunk_buffer_conf['sample_rate'],
        sample_width=chunk_buffer_conf['sample_width'])
    # init vad
    vad_conf = asr_engine.config.vad_conf
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
                    await websocket.send_json(resp)
                elif message['signal'] == 'end':
                    engine_pool = get_engine_pool()
                    asr_engine = engine_pool['asr']
                    # reset single  engine for an new connection
                    asr_engine.reset()
                    resp = {"status": "ok", "signal": "finished"}
                    await websocket.send_json(resp)
                    break
                else:
                    resp = {"status": "ok", "message": "no valid json data"}
                    await websocket.send_json(resp)
            elif "bytes" in message:
                message = message["bytes"]

                # vad for input bytes audio
                vad.add_audio(message)
                message = b''.join(f for f in vad.vad_collector()
                                   if f is not None)

                engine_pool = get_engine_pool()
                asr_engine = engine_pool['asr']
                asr_results = ""
                frames = chunk_buffer.frame_generator(message)
                for frame in frames:
                    samples = np.frombuffer(frame.bytes, dtype=np.int16)
                    sample_rate = asr_engine.config.sample_rate
                    x_chunk, x_chunk_lens = asr_engine.preprocess(samples,
                                                                  sample_rate)
                    asr_engine.run(x_chunk, x_chunk_lens)
                    asr_results = asr_engine.postprocess()

                asr_results = asr_engine.postprocess()
                resp = {'asr_results': asr_results}

                await websocket.send_json(resp)
    except WebSocketDisconnect:
        pass
