import axios from 'axios'
import {apiURL} from "./API.js"

// 获取闲聊对话结果
export async function nlpChat(text){
    const result = await axios.post(apiURL.NLP_CHAT, { chat : text});
    return result
}

// 获取信息抽取结果
export async function nlpIE(text){
    const result = await axios.post(apiURL.NLP_IE, { chat : text});
    return result
}



