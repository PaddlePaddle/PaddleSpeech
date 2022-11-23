import axios from 'axios'
import {apiURL} from "./API.js"

export async function ttsOffline(text){
    const result = await axios.post(apiURL.TTS_OFFLINE, { text : text});
    return result
}

