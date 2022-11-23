import axios from 'axios'
import {apiURL} from "./API.js"

// 注册声纹
export async function vprEnroll(params){
    const result = await axios.post(apiURL.VPR_ENROLL, params);
    return result
}

// 声纹识别
export async function vprRecog(params){
    const result = await axios.post(apiURL.VPR_RECOG, params);
    return result
}

// 删除声纹
export async function vprDel(params){
    const result = await axios.post(apiURL.VPR_DEL, params);
    return result
}

// 获取声纹列表
export async function vprList(){
    const result = await axios.get(apiURL.VPR_LIST);
    return result
}

// 获取声纹音频
export async function vprData(params){
    const result = await axios.get(apiURL.VPR_DATA+params);
    return result
}
