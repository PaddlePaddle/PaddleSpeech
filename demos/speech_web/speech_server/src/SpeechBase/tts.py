# tts 推理引擎，支持流式与非流式
# 精简化使用
# 用 onnxruntime 进行推理
# 1. 下载对应的模型
# 2. 加载模型
# 3. 端到端推理
# 4. 流式推理

import base64

import numpy as np
from paddlespeech.server.utils.onnx_infer import get_sess
from paddlespeech.t2s.frontend.zh_frontend import Frontend
from paddlespeech.server.utils.util import denorm, get_chunks
from paddlespeech.server.utils.audio_process import float2pcm
from paddlespeech.server.utils.config import get_config

from paddlespeech.server.engine.tts.online.onnx.tts_engine import TTSEngine


class TTS:
    def __init__(self, config_path):
        self.config = get_config(config_path)['tts_online-onnx']
        self.config['voc_block'] = 36
        self.engine =  TTSEngine()
        self.engine.init(self.config)
        self.engine.warm_up()
        
        # 前端初始化
        self.frontend = Frontend(
                phone_vocab_path=self.engine.executor.phones_dict,
                tone_vocab_path=None)
    
    def depadding(self, data, chunk_num, chunk_id, block, pad, upsample):
        """ 
        Streaming inference removes the result of pad inference
        """
        front_pad = min(chunk_id * block, pad)
        # first chunk
        if chunk_id == 0:
            data = data[:block * upsample]
        # last chunk
        elif chunk_id == chunk_num - 1:
            data = data[front_pad * upsample:]
        # middle chunk
        else:
            data = data[front_pad * upsample:(front_pad + block) * upsample]

        return data
         
    def offlineTTS(self, text):
        get_tone_ids = False
        merge_sentences = False
        
        input_ids = self.frontend.get_input_ids(
                text,
                merge_sentences=merge_sentences,
                get_tone_ids=get_tone_ids)
        phone_ids = input_ids["phone_ids"]
        wav_list = []
        for i in range(len(phone_ids)):
            orig_hs = self.engine.executor.am_encoder_infer_sess.run(
                            None, input_feed={'text': phone_ids[i].numpy()}
                            )
            hs = orig_hs[0]
            am_decoder_output = self.engine.executor.am_decoder_sess.run(
                        None, input_feed={'xs': hs})
            am_postnet_output = self.engine.executor.am_postnet_sess.run(
                        None,
                        input_feed={
                            'xs': np.transpose(am_decoder_output[0], (0, 2, 1))
                        })
            am_output_data = am_decoder_output + np.transpose(
                am_postnet_output[0], (0, 2, 1))
            normalized_mel = am_output_data[0][0]
            mel = denorm(normalized_mel, self.engine.executor.am_mu, self.engine.executor.am_std)
            wav = self.engine.executor.voc_sess.run(
                            output_names=None, input_feed={'logmel': mel})[0]
            wav_list.append(wav)
        wavs = np.concatenate(wav_list)
        return wavs
    
    def streamTTS(self, text):
        for sub_wav_base64 in self.engine.run(sentence=text):
            yield sub_wav_base64
    
    def streamTTSBytes(self, text):
        for wav in self.engine.executor.infer(
                text=text,
                lang=self.engine.config.lang,
                am=self.engine.config.am,
                spk_id=0):
            wav = float2pcm(wav)  # float32 to int16
            wav_bytes = wav.tobytes()  # to bytes
            yield wav_bytes
        
    
    def after_process(self, wav):
        # for tvm
        wav = float2pcm(wav)  # float32 to int16
        wav_bytes = wav.tobytes()  # to bytes
        wav_base64 = base64.b64encode(wav_bytes).decode('utf8')  # to base64
        return wav_base64
    
    def streamTTS_TVM(self, text):
        # 用 TVM 优化
        pass

if __name__ == '__main__':
    text = "啊哈哈哈哈哈哈啊哈哈哈哈哈哈啊哈哈哈哈哈哈啊哈哈哈哈哈哈啊哈哈哈哈哈哈"
    config_path="../../PaddleSpeech/demos/streaming_tts_server/conf/tts_online_application.yaml"
    tts = TTS(config_path)
    
    for sub_wav in tts.streamTTS(text):
        print("sub_wav_base64: ", len(sub_wav))
    
    end_wav = tts.offlineTTS(text)
    print(end_wav)
    
    
        