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
from fastapi import UploadFile
from pydantic import BaseModel
from src.ernie_sat import SAT
from src.finetune import FineTune
from src.ge2e_clone import VoiceCloneGE2E
from src.tdnn_clone import VoiceCloneTDNN
from src.util import *
from starlette.responses import FileResponse

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

# 这里会对finetune产生影响，所以finetune使用了cmd
vc_model = VoiceCloneGE2E()
vc_model_tdnn = VoiceCloneTDNN()

sat_model = SAT()
ft_model = FineTune()

# 配置文件
tts_config = "conf/tts_online_application.yaml"
asr_config = "conf/ws_conformer_wenetspeech_application_faster.yaml"
asr_init_path = "source/demo/demo.wav"
db_path = "source/db/vc.sqlite"
ie_model_path = "source/model"

# 路径配置
VC_UPLOAD_PATH = "source/wav/vc/upload"
VC_OUT_PATH = "source/wav/vc/out"

FT_UPLOAD_PATH = "source/wav/finetune/upload"
FT_OUT_PATH = "source/wav/finetune/out"
FT_LABEL_PATH = "source/wav/finetune/label.json"
FT_LABEL_TXT_PATH = "source/wav/finetune/labels.txt"
FT_DEFAULT_PATH = "source/wav/finetune/default"
FT_EXP_BASE_PATH = "tmp_dir/finetune"

SAT_UPLOAD_PATH = "source/wav/SAT/upload"
SAT_OUT_PATH = "source/wav/SAT/out"
SAT_LABEL_PATH = "source/wav/SAT/label.json"

# SAT 标注结果初始化
if os.path.exists(SAT_LABEL_PATH):
    with open(SAT_LABEL_PATH, "r", encoding='utf8') as f:
        sat_label_dic = json.load(f)
else:
    sat_label_dic = {}

# ft 标注结果初始化
if os.path.exists(FT_LABEL_PATH):
    with open(FT_LABEL_PATH, "r", encoding='utf8') as f:
        ft_label_dic = json.load(f)
else:
    ft_label_dic = {}

# 新建文件夹
base_sources = [
    VC_UPLOAD_PATH,
    VC_OUT_PATH,
    FT_UPLOAD_PATH,
    FT_OUT_PATH,
    FT_DEFAULT_PATH,
    SAT_UPLOAD_PATH,
    SAT_OUT_PATH,
]
for path in base_sources:
    os.makedirs(path, exist_ok=True)
#####################################################################
########################### APP初始化  ###############################
#####################################################################
app = FastAPI()

######################################################################
########################### 接口类型  #################################
#####################################################################


# 接口结构
class VcBase(BaseModel):
    wavName: str
    wavPath: str


class VcBaseText(BaseModel):
    wavName: str
    wavPath: str
    text: str
    func: str


class VcBaseSAT(BaseModel):
    old_str: str
    new_str: str
    language: str
    function: str
    wav: str  # base64编码
    filename: str


class FTPath(BaseModel):
    dataPath: str


class VcBaseFT(BaseModel):
    wav: str  # base64编码
    filename: str
    wav_path: str


class VcBaseFTModel(BaseModel):
    wav_path: str


class VcBaseFTSyn(BaseModel):
    exp_path: str
    text: str


######################################################################
########################### 文件列表查询与保存服务 #################################
#####################################################################


def getVCList(path):
    VC_FileDict = []
    # 查询upload路径下的wav文件名
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            # print(os.path.join(root, name))
            VC_FileDict.append({'name': name, 'path': os.path.join(root, name)})
    VC_FileDict = sorted(VC_FileDict, key=lambda x: x['name'], reverse=True)
    return VC_FileDict


async def saveFiles(files, SavePath):
    right = 0
    error = 0
    error_info = "错误文件："
    for file in files:
        try:
            if 'blob' in file.filename:
                out_file_path = os.path.join(
                    SavePath,
                    datetime.datetime.strftime(datetime.datetime.now(),
                                               '%H%M') + randName(3) + ".wav")
            else:
                out_file_path = os.path.join(SavePath, file.filename)

            print("上传文件名:", out_file_path)
            async with aiofiles.open(out_file_path, 'wb') as out_file:
                content = await file.read()  # async read
                await out_file.write(content)  # async write
            # 将文件转成24k, 16bit类型的wav文件
            wav, sr = librosa.load(out_file_path, sr=16000)
            sf.write(out_file_path, data=wav, samplerate=sr)
            right += 1
        except Exception as e:
            error += 1
            error_info = error_info + file.filename + " " + str(e) + "\n"
            continue
    return f"上传成功：{right}, 上传失败：{error}, 失败原因： {error_info}"


# 音频下载
@app.post("/vc/download")
async def VcDownload(base: VcBase):
    if os.path.exists(base.wavPath):
        return FileResponse(base.wavPath)
    else:
        return ErrorRequest(message="下载请求失败，文件不存在")


# 音频下载base64
@app.post("/vc/download_base64")
async def VcDownloadBase64(base: VcBase):
    if os.path.exists(base.wavPath):
        # 将文件转成16k, 16bit类型的wav文件
        wav, sr = librosa.load(base.wavPath, sr=16000)
        wav = float2pcm(wav)  # float32 to int16
        wav_bytes = wav.tobytes()  # to bytes
        wav_base64 = base64.b64encode(wav_bytes).decode('utf8')
        return SuccessRequest(result=wav_base64)
    else:
        return ErrorRequest(message="播放请求失败，文件不存在")


######################################################################
########################### VC 服务 #################################
#####################################################################


# 上传文件
@app.post("/vc/upload")
async def VcUpload(files: List[UploadFile]):
    # res = saveFiles(files, VC_UPLOAD_PATH)
    right = 0
    error = 0
    error_info = "错误文件："
    for file in files:
        try:
            if 'blob' in file.filename:
                out_file_path = os.path.join(
                    VC_UPLOAD_PATH,
                    datetime.datetime.strftime(datetime.datetime.now(),
                                               '%H%M') + randName(3) + ".wav")
            else:
                out_file_path = os.path.join(VC_UPLOAD_PATH, file.filename)

            print("上传文件名:", out_file_path)
            async with aiofiles.open(out_file_path, 'wb') as out_file:
                content = await file.read()  # async read
                await out_file.write(content)  # async write
            # 将文件转成24k, 16bit类型的wav文件
            wav, sr = librosa.load(out_file_path, sr=16000)
            sf.write(out_file_path, data=wav, samplerate=sr)
            right += 1
        except Exception as e:
            error += 1
            error_info = error_info + file.filename + " " + str(e) + "\n"
            continue
    return SuccessRequest(
        result=f"上传成功：{right}, 上传失败：{error}, 失败原因： {error_info}")


# 获取文件列表
@app.get("/vc/list")
async def VcList():
    res = getVCList(VC_UPLOAD_PATH)
    return SuccessRequest(result=res)


# 获取音频文件
@app.post("/vc/file")
async def VcFileGet(base: VcBase):
    if os.path.exists(base.wavPath):
        return FileResponse(base.wavPath)
    else:
        return ErrorRequest(result="获取文件失败")


# 删除音频文件
@app.post("/vc/del")
async def VcFileDel(base: VcBase):
    if os.path.exists(base.wavPath):
        os.remove(base.wavPath)
        return SuccessRequest(result="删除成功")
    else:
        return ErrorRequest(result="删除失败")


# 声音克隆G2P
@app.post("/vc/clone_g2p")
async def VcCloneG2P(base: VcBaseText):
    if os.path.exists(base.wavPath):
        try:
            if base.func == 'ge2e':
                wavName = base.wavName
                wavPath = os.path.join(VC_OUT_PATH, wavName)
                wavPath = vc_model.vc(
                    text=base.text, input_wav=base.wavPath, out_wav=wavPath)
            else:
                wavName = base.wavName
                wavPath = os.path.join(VC_OUT_PATH, wavName)
                wavPath = vc_model_tdnn.vc(
                    text=base.text, input_wav=base.wavPath, out_wav=wavPath)
            if wavPath:
                res = {"wavName": wavName, "wavPath": wavPath}
                return SuccessRequest(result=res)
            else:
                return ErrorRequest(message="克隆失败，检查克隆脚本是否有效")
        except Exception as e:
            print(e)
            return ErrorRequest(message="克隆失败，合成过程报错")
    else:
        return ErrorRequest(message="克隆失败，音频不存在")


######################################################################
########################### SAT 服务 #################################
#####################################################################
# 声音克隆SAT
@app.post("/vc/clone_sat")
async def VcCloneSAT(base: VcBaseSAT):
    # 重新整理 sat_label_dict
    if base.filename not in sat_label_dic or sat_label_dic[
            base.filename] != base.old_str:
        sat_label_dic[base.filename] = base.old_str
        with open(SAT_LABEL_PATH, "w", encoding='utf8') as f:
            json.dump(sat_label_dic, f, ensure_ascii=False, indent=4)

    input_file_path = base.wav

    # 选择任务
    if base.language == "zh":
        # 中文
        if base.function == "synthesize":
            output_file_path = os.path.join(SAT_OUT_PATH,
                                            "sat_syn_zh_" + base.filename)
            # 中文克隆
            sat_result = sat_model.zh_synthesize_edit(
                old_str=base.old_str,
                new_str=base.new_str,
                input_name=os.path.realpath(input_file_path),
                output_name=os.path.realpath(output_file_path),
                task_name="synthesize")
        elif base.function == "edit":
            output_file_path = os.path.join(SAT_OUT_PATH,
                                            "sat_edit_zh_" + base.filename)
            # 中文语音编辑
            sat_result = sat_model.zh_synthesize_edit(
                old_str=base.old_str,
                new_str=base.new_str,
                input_name=os.path.realpath(input_file_path),
                output_name=os.path.realpath(output_file_path),
                task_name="edit")
        elif base.function == "crossclone":
            output_file_path = os.path.join(SAT_OUT_PATH,
                                            "sat_cross_zh_" + base.filename)
            # 中文跨语言
            sat_result = sat_model.crossclone(
                old_str=base.old_str,
                new_str=base.new_str,
                input_name=os.path.realpath(input_file_path),
                output_name=os.path.realpath(output_file_path),
                source_lang="zh",
                target_lang="en")
        else:
            return ErrorRequest(
                message="请检查功能选项是否正确，仅支持:synthesize, edit, crossclone")
    elif base.language == "en":
        if base.function == "synthesize":
            output_file_path = os.path.join(SAT_OUT_PATH,
                                            "sat_syn_zh_" + base.filename)
            # 英文语音克隆
            sat_result = sat_model.en_synthesize_edit(
                old_str=base.old_str,
                new_str=base.new_str,
                input_name=os.path.realpath(input_file_path),
                output_name=os.path.realpath(output_file_path),
                task_name="synthesize")
        elif base.function == "edit":
            output_file_path = os.path.join(SAT_OUT_PATH,
                                            "sat_edit_zh_" + base.filename)
            # 英文语音编辑
            sat_result = sat_model.en_synthesize_edit(
                old_str=base.old_str,
                new_str=base.new_str,
                input_name=os.path.realpath(input_file_path),
                output_name=os.path.realpath(output_file_path),
                task_name="edit")
        elif base.function == "crossclone":
            output_file_path = os.path.join(SAT_OUT_PATH,
                                            "sat_cross_zh_" + base.filename)
            # 英文跨语言
            sat_result = sat_model.crossclone(
                old_str=base.old_str,
                new_str=base.new_str,
                input_name=os.path.realpath(input_file_path),
                output_name=os.path.realpath(output_file_path),
                source_lang="en",
                target_lang="zh")
        else:
            return ErrorRequest(
                message="请检查功能选项是否正确，仅支持:synthesize, edit, crossclone")
    else:
        return ErrorRequest(message="请检查功能选项是否正确，仅支持中文和英文")

    if sat_result:
        return SuccessRequest(result=sat_result, message="SAT合成成功")
    else:
        return ErrorRequest(message="SAT 合成失败，请从后台检查错误信息！")


# SAT 文件列表
@app.get("/sat/list")
async def SatList():
    res = []
    filelist = getVCList(SAT_UPLOAD_PATH)
    for fileitem in filelist:
        if fileitem['name'] in sat_label_dic:
            fileitem['label'] = sat_label_dic[fileitem['name']]
        else:
            fileitem['label'] = ""
        res.append(fileitem)
    return SuccessRequest(result=res)


# 上传 SAT 音频
# 上传文件
@app.post("/sat/upload")
async def SATUpload(files: List[UploadFile]):
    right = 0
    error = 0
    error_info = "错误文件："
    for file in files:
        try:
            if 'blob' in file.filename:
                out_file_path = os.path.join(
                    SAT_UPLOAD_PATH,
                    datetime.datetime.strftime(datetime.datetime.now(),
                                               '%H%M') + randName(3) + ".wav")
            else:
                out_file_path = os.path.join(SAT_UPLOAD_PATH, file.filename)

            print("上传文件名:", out_file_path)
            async with aiofiles.open(out_file_path, 'wb') as out_file:
                content = await file.read()  # async read
                await out_file.write(content)  # async write
            # 将文件转成24k, 16bit类型的wav文件
            wav, sr = librosa.load(out_file_path, sr=16000)
            sf.write(out_file_path, data=wav, samplerate=sr)
            right += 1
        except Exception as e:
            error += 1
            error_info = error_info + file.filename + " " + str(e) + "\n"
            continue
    return SuccessRequest(
        result=f"上传成功：{right}, 上传失败：{error}, 失败原因： {error_info}")


######################################################################
########################### FinueTune 服务 #################################
#####################################################################


# finetune 文件列表
@app.post("/finetune/list")
async def FineTuneList(Path: FTPath):
    dataPath = Path.dataPath
    if dataPath == "default":
        # 默认路径
        FT_PATH = FT_DEFAULT_PATH
    else:
        FT_PATH = dataPath

    res = []
    filelist = getVCList(FT_PATH)
    for name, value in ft_label_dic.items():
        wav_path = os.path.join(FT_PATH, name)
        if not os.path.exists(wav_path):
            wav_path = ""
        d = {'text': value['text'], 'name': name, 'path': wav_path}
        res.append(d)
    return SuccessRequest(result=res)


# 一键重置，获取新的文件地址
@app.get('/finetune/newdir')
async def FTGetNewDir():
    new_path = os.path.join(FT_UPLOAD_PATH, randName(3))
    if not os.path.exists(new_path):
        os.makedirs(new_path, exist_ok=True)
    # 把 labels.txt 复制进去
    cmd = f"cp {FT_LABEL_TXT_PATH} {new_path}"
    os.system(cmd)
    return SuccessRequest(result=new_path)


# finetune 上传文件
@app.post("/finetune/upload")
async def FTUpload(base: VcBaseFT):
    try:
        # 文件夹是否存在
        if not os.path.exists(base.wav_path):
            os.makedirs(base.wav_path)
        # 保存音频文件
        out_file_path = os.path.join(base.wav_path, base.filename)
        wav_b = base64.b64decode(base.wav)
        async with aiofiles.open(out_file_path, 'wb') as out_file:
            await out_file.write(wav_b)  # async write

        return SuccessRequest(result="上传成功")
    except Exception as e:
        return ErrorRequest(result="上传失败")


# finetune 微调
@app.post("/finetune/clone_finetune")
async def FTModel(base: VcBaseFTModel):
    # 先检查 wav_path 是否有效
    if base.wav_path == 'default':
        data_path = FT_DEFAULT_PATH
    else:
        data_path = base.wav_path
    if not os.path.exists(data_path):
        return ErrorRequest(message="数据文件夹不存在")

    data_base = data_path.split(os.sep)[-1]
    exp_dir = os.path.join(FT_EXP_BASE_PATH, data_base)
    try:
        exp_dir = ft_model.finetune(
            input_dir=os.path.realpath(data_path),
            exp_dir=os.path.realpath(exp_dir))
        if exp_dir:
            return SuccessRequest(result=exp_dir)
        else:
            return ErrorRequest(message="微调失败")
    except Exception as e:
        print(e)
        return ErrorRequest(message="微调失败")


# finetune 合成
@app.post("/finetune/clone_finetune_syn")
async def FTSyn(base: VcBaseFTSyn):
    try:
        if not os.path.exists(base.exp_path):
            return ErrorRequest(result="模型路径不存在")
        wav_name = randName(5)
        wav_path = ft_model.synthesize(
            text=base.text,
            wav_name=wav_name,
            out_wav_dir=os.path.realpath(FT_OUT_PATH),
            exp_dir=os.path.realpath(base.exp_path))
        if wav_path:
            res = {"wavName": wav_name + ".wav", "wavPath": wav_path}
            return SuccessRequest(result=res)
        else:
            return ErrorRequest(message="音频合成失败")
    except Exception as e:
        return ErrorRequest(message="音频合成失败")


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=port)
