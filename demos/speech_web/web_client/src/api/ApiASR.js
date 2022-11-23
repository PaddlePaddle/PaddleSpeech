import axios from 'axios'
import {apiURL} from "./API.js"

// 上传音频文件，获得识别结果
export async function asrOffline(params){
    const result = await axios.post(
        apiURL.ASR_OFFLINE, params
    )
    return result
}

// 上传环境采集文件
export async function asrCollentEnv(params){
    const result = await axios.post(
        apiURL.ASR_OFFLINE, params
    )
    return result
}

// 暂停录音
export async function asrStopRecord(){
    const result = await axios.get(apiURL.ASR_STOP_RECORD);
    return result
}

// 恢复录音
export async function asrResumeRecord(){
    const result = await axios.get(apiURL.ASR_RESUME_RECORD);
    return result
}