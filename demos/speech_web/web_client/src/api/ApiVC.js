import axios from 'axios'
import {apiURL} from "./API.js"

// 上传音频-vc
export async function vcUpload(params){
    const result = await axios.post(apiURL.VC_Upload, params);
    return result
}

// 上传音频-sat
export async function satUpload(params){
    const result = await axios.post(apiURL.SAT_Upload, params);
    return result
}

// 上传音频-finetune
export async function fineTuneUpload(params){
    const result = await axios.post(apiURL.FineTune_Upload, params);
    return result
}

// 删除音频
export async function vcDel(params){
    const result = await axios.post(apiURL.VC_Del, params);
    return result
}

// 获取音频列表vc
export async function vcList(){
    const result = await axios.get(apiURL.VC_List);
    return result
}
// 获取音频列表Sat
export async function satList(){
    const result = await axios.get(apiURL.SAT_List);
    return result
}

// 获取音频列表fineTune
export async function fineTuneList(params){
    const result = await axios.post(apiURL.FineTune_List, params);
    return result
}

// fineTune 一键重置 获取新的文件夹
export async function fineTuneNewDir(){
    const result = await axios.get(apiURL.FineTune_NewDir);
    return result
}

// 获取音频数据
export async function vcDownload(params){
    const result = await axios.post(apiURL.VC_Download, params);
    return result
}

// 获取音频数据Base64
export async function vcDownloadBase64(params){
    const result = await axios.post(apiURL.VC_Download_Base64, params);
    return result
}


// 克隆合成G2P
export async function vcCloneG2P(params){
    const result = await axios.post(apiURL.VC_CloneG2p, params);
    return result
}

// 克隆合成SAT
export async function vcCloneSAT(params){
    const result = await axios.post(apiURL.VC_CloneSAT, params);
    return result
}

// 克隆合成 - finetune 微调
export async function vcCloneFineTune(params){
    const result = await axios.post(apiURL.VC_CloneFineTune, params);
    return result
}

// 克隆合成 - finetune 合成
export async function vcCloneFineTuneSyn(params){
    const result = await axios.post(apiURL.VC_CloneFineTuneSyn, params);
    return result
}


