import argparse
import base64
import datetime
import json
import os
from typing import List

import aiofiles
import librosa
import soundfile as sf
import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import UploadFile
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.AudioManeger import AudioMannger
from src.robot import Robot
from src.SpeechBase.vpr import VPR
from src.util import *
from src.WebsocketManeger import ConnectionManager
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse
from starlette.websockets import WebSocketState as WebSocketState

from paddlespeech.cli.tts.infer import TTSExecutor
from paddlespeech.server.engine.asr.online.python.asr_engine import PaddleASRConnectionHanddler
from paddlespeech.server.utils.audio_process import float2pcm

# 解析配置
parser = argparse.ArgumentParser(prog='PaddleSpeechDemo', add_help=True)

parser.add_argument(
    "--port",
    action="store",
    type=int,
    help="port of the app",
    default=8010,
    required=False)

args = parser.parse_args()
port = args.port

# 配置文件
tts_config = "conf/tts_online_application.yaml"
asr_config = "conf/ws_conformer_wenetspeech_application_faster.yaml"
asr_init_path = "source/demo/demo.wav"
db_path = "source/db/vpr.sqlite"
ie_model_path = "source/model"
tts_model = TTSExecutor()
# 路径配置
UPLOAD_PATH = "source/vpr"
WAV_PATH = "source/wav"

base_sources = [UPLOAD_PATH, WAV_PATH]
for path in base_sources:
    os.makedirs(path, exist_ok=True)

# 初始化
app = FastAPI()
chatbot = Robot(
    asr_config, tts_config, asr_init_path, ie_model_path=ie_model_path)
manager = ConnectionManager()
aumanager = AudioMannger(chatbot)
aumanager.init()
vpr = VPR(db_path, dim=192, top_k=5)
# 初始化下载模型
tts_model(
    text="今天天气准不错",
    output="test.wav",
    am='fastspeech2_mix',
    spk_id=174,
    voc='hifigan_csmsc',
    lang='mix', )


# 服务配置
class NlpBase(BaseModel):
    chat: str


class TtsBase(BaseModel):
    text: str


class Audios:
    def __init__(self) -> None:
        self.audios = b""


audios = Audios()

######################################################################
########################### ASR 服务 #################################
#####################################################################


# 接收文件，返回ASR结果
# 上传文件
@app.post("/asr/offline")
async def speech2textOffline(files: List[UploadFile]):
    # 只有第一个有效
    asr_res = ""
    for file in files[:1]:
        # 生成时间戳
        now_name = "asr_offline_" + datetime.datetime.strftime(
            datetime.datetime.now(), '%Y%m%d%H%M%S') + randName() + ".wav"
        out_file_path = os.path.join(WAV_PATH, now_name)
        async with aiofiles.open(out_file_path, 'wb') as out_file:
            content = await file.read()  # async read
            await out_file.write(content)  # async write

        # 返回ASR识别结果
        asr_res = chatbot.speech2text(out_file_path)
        return SuccessRequest(result=asr_res)
    return ErrorRequest(message="上传文件为空")


# 接收文件，同时将wav强制转成16k, int16类型
@app.post("/asr/offlinefile")
async def speech2textOfflineFile(files: List[UploadFile]):
    # 只有第一个有效
    asr_res = ""
    for file in files[:1]:
        # 生成时间戳
        now_name = "asr_offline_" + datetime.datetime.strftime(
            datetime.datetime.now(), '%Y%m%d%H%M%S') + randName() + ".wav"
        out_file_path = os.path.join(WAV_PATH, now_name)
        async with aiofiles.open(out_file_path, 'wb') as out_file:
            content = await file.read()  # async read
            await out_file.write(content)  # async write

        # 将文件转成16k, 16bit类型的wav文件
        wav, sr = librosa.load(out_file_path, sr=16000)
        wav = float2pcm(wav)  # float32 to int16
        wav_bytes = wav.tobytes()  # to bytes
        wav_base64 = base64.b64encode(wav_bytes).decode('utf8')

        # 将文件重新写入
        now_name = now_name[:-4] + "_16k" + ".wav"
        out_file_path = os.path.join(WAV_PATH, now_name)
        sf.write(out_file_path, wav, 16000)

        # 返回ASR识别结果
        asr_res = chatbot.speech2text(out_file_path)
        response_res = {"asr_result": asr_res, "wav_base64": wav_base64}
        return SuccessRequest(result=response_res)

    return ErrorRequest(message="上传文件为空")


# 流式接收测试
@app.post("/asr/online1")
async def speech2textOnlineRecive(files: List[UploadFile]):
    audio_bin = b''
    for file in files:
        content = await file.read()
        audio_bin += content
    audios.audios += audio_bin
    print(f"audios长度变化: {len(audios.audios)}")
    return SuccessRequest(message="接收成功")


# 采集环境噪音大小
@app.post("/asr/collectEnv")
async def collectEnv(files: List[UploadFile]):
    for file in files[:1]:
        content = await file.read()  # async read
        # 初始化, wav 前44字节是头部信息
        aumanager.compute_env_volume(content[44:])
        vad_ = aumanager.vad_threshold
        return SuccessRequest(result=vad_, message="采集环境噪音成功")


# 停止录音
@app.get("/asr/stopRecord")
async def stopRecord():
    audios.audios = b""
    aumanager.stop()
    print("Online录音暂停")
    return SuccessRequest(message="停止成功")


# 恢复录音
@app.get("/asr/resumeRecord")
async def resumeRecord():
    aumanager.resume()
    print("Online录音恢复")
    return SuccessRequest(message="Online录音恢复")


# 聊天用的 ASR
@app.websocket("/ws/asr/offlineStream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            asr_res = None
            # websocket 不接收，只推送
            data = await websocket.receive_bytes()
            if not aumanager.is_pause:
                asr_res = aumanager.stream_asr(data)
            else:
                print("录音暂停")
            if asr_res:
                await manager.send_personal_message(asr_res, websocket)
                aumanager.clear_asr()

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        # await manager.broadcast(f"用户-{user}-离开")
        # print(f"用户-{user}-离开")


    # 流式识别的 ASR
@app.websocket('/ws/asr/onlineStream')
async def websocket_endpoint_online(websocket: WebSocket):
    """PaddleSpeech Online ASR Server api

    Args:
        websocket (WebSocket): the websocket instance
    """

    #1. the interface wait to accept the websocket protocal header
    #   and only we receive the header, it establish the connection with specific thread
    await websocket.accept()

    #2. if we accept the websocket headers, we will get the online asr engine instance
    engine = chatbot.asr.engine

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
                    # do something at beginning here
                    # create the instance to process the audio
                    # connection_handler = chatbot.asr.connection_handler
                    connection_handler = PaddleASRConnectionHanddler(engine)
                    await websocket.send_json(resp)
                elif message['signal'] == 'end':
                    # reset single  engine for an new connection
                    # and we will destroy the connection
                    connection_handler.decode(is_finished=True)
                    connection_handler.rescoring()
                    asr_results = connection_handler.get_result()
                    connection_handler.reset()

                    resp = {
                        "status": "ok",
                        "signal": "finished",
                        'result': asr_results
                    }
                    await websocket.send_json(resp)
                    break
                else:
                    resp = {"status": "ok", "message": "no valid json data"}
                    await websocket.send_json(resp)
            elif "bytes" in message:
                # bytes for the pcm data
                message = message["bytes"]
                print("###############")
                print("len message: ", len(message))
                print("###############")

                # we extract the remained audio pcm 
                # and decode for the result in this package data
                connection_handler.extract_feat(message)
                connection_handler.decode(is_finished=False)
                asr_results = connection_handler.get_result()

                # return the current period result
                # if the engine create the vad instance, this connection will have many period results 
                resp = {'result': asr_results}
                print(resp)
                await websocket.send_json(resp)
    except WebSocketDisconnect:
        pass


######################################################################
########################### NLP 服务 #################################
#####################################################################


@app.post("/nlp/chat")
async def chatOffline(nlp_base: NlpBase):
    chat = nlp_base.chat
    if not chat:
        return ErrorRequest(message="传入文本为空")
    else:
        res = chatbot.chat(chat)
        return SuccessRequest(result=res)


@app.post("/nlp/ie")
async def ieOffline(nlp_base: NlpBase):
    nlp_text = nlp_base.chat
    if not nlp_text:
        return ErrorRequest(message="传入文本为空")
    else:
        res = chatbot.ie(nlp_text)
        return SuccessRequest(result=res)


######################################################################
########################### TTS 服务 #################################
#####################################################################


# 端到端合成
@app.post("/tts/offline")
async def text2speechOffline(tts_base: TtsBase):
    text = tts_base.text
    if not text:
        return ErrorRequest(message="文本为空")
    else:
        now_name = "tts_" + datetime.datetime.strftime(
            datetime.datetime.now(), '%Y%m%d%H%M%S') + randName() + ".wav"
        out_file_path = os.path.join(WAV_PATH, now_name)
        # 使用中英混合CLI
        tts_model(
            text=text,
            output=out_file_path,
            am='fastspeech2_mix',
            spk_id=174,
            voc='hifigan_csmsc',
            lang='mix')
        with open(out_file_path, "rb") as f:
            data_bin = f.read()
        base_str = base64.b64encode(data_bin)
        return SuccessRequest(result=base_str)


# http流式TTS
@app.post("/tts/online")
async def stream_tts(request_body: TtsBase):
    text = request_body.text
    return StreamingResponse(chatbot.text2speechStreamBytes(text=text))


# ws流式TTS
@app.websocket("/ws/tts/online")
async def stream_ttsWS(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            text = await websocket.receive_text()
            # 用 websocket 流式接收音频数据
            if text:
                for sub_wav in chatbot.text2speechStream(text=text):
                    # print("发送sub wav: ", len(sub_wav))
                    res = {"wav": sub_wav, "done": False}
                    await websocket.send_json(res)

                # 输送结束
                res = {"wav": sub_wav, "done": True}
                await websocket.send_json(res)
            # manager.disconnect(websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)


######################################################################
########################### VPR 服务 #################################
#####################################################################

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


@app.post('/vpr/enroll')
async def vpr_enroll(table_name: str=None,
                     spk_id: str=Form(...),
                     audio: UploadFile=File(...)):
    # Enroll the uploaded audio with spk-id into MySQL
    try:
        if not spk_id:
            return {'status': False, 'msg': "spk_id can not be None"}
        # Save the upload data to server.
        content = await audio.read()
        now_name = "vpr_enroll_" + datetime.datetime.strftime(
            datetime.datetime.now(), '%Y%m%d%H%M%S') + randName() + ".wav"
        audio_path = os.path.join(UPLOAD_PATH, now_name)

        with open(audio_path, "wb+") as f:
            f.write(content)
        vpr.vpr_enroll(username=spk_id, wav_path=audio_path)
        return {'status': True, 'msg': "Successfully enroll data!"}
    except Exception as e:
        return {'status': False, 'msg': e}


@app.post('/vpr/recog')
async def vpr_recog(request: Request,
                    table_name: str=None,
                    audio: UploadFile=File(...)):
    # Voice print recognition online
    # try:
    # Save the upload data to server.
    content = await audio.read()
    now_name = "vpr_query_" + datetime.datetime.strftime(
        datetime.datetime.now(), '%Y%m%d%H%M%S') + randName() + ".wav"
    query_audio_path = os.path.join(UPLOAD_PATH, now_name)
    with open(query_audio_path, "wb+") as f:
        f.write(content)
    spk_ids, paths, scores = vpr.do_search_vpr(query_audio_path)

    res = dict(zip(spk_ids, zip(paths, scores)))
    # Sort results by distance metric, closest distances first
    res = sorted(res.items(), key=lambda item: item[1][1], reverse=True)
    return res


@app.post('/vpr/del')
async def vpr_del(spk_id: dict=None):
    # Delete a record by spk_id in MySQL
    try:
        spk_id = spk_id['spk_id']
        if not spk_id:
            return {'status': False, 'msg': "spk_id can not be None"}
        vpr.vpr_del(username=spk_id)
        return {'status': True, 'msg': "Successfully delete data!"}
    except Exception as e:
        return {'status': False, 'msg': e}, 400


@app.get('/vpr/list')
async def vpr_list():
    # Get all records in MySQL
    try:
        spk_ids, vpr_ids = vpr.do_list()
        return spk_ids, vpr_ids
    except Exception as e:
        return {'status': False, 'msg': e}, 400


@app.get('/vpr/database64')
async def vpr_database64(vprId: int):
    # Get the audio file from path by spk_id in MySQL
    try:
        if not vprId:
            return {'status': False, 'msg': "vpr_id can not be None"}
        audio_path = vpr.do_get_wav(vprId)
        # 返回base64

        # 将文件转成16k, 16bit类型的wav文件
        wav, sr = librosa.load(audio_path, sr=16000)
        wav = float2pcm(wav)  # float32 to int16
        wav_bytes = wav.tobytes()  # to bytes
        wav_base64 = base64.b64encode(wav_bytes).decode('utf8')

        return SuccessRequest(result=wav_base64)
    except Exception as e:
        return {'status': False, 'msg': e}, 400


@app.get('/vpr/data')
async def vpr_data(vprId: int):
    # Get the audio file from path by spk_id in MySQL
    try:
        if not vprId:
            return {'status': False, 'msg': "vpr_id can not be None"}
        audio_path = vpr.do_get_wav(vprId)
        return FileResponse(audio_path)
    except Exception as e:
        return {'status': False, 'msg': e}, 400


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=port)
