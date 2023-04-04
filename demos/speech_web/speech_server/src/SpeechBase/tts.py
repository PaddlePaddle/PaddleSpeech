# tts 推理引擎，支持流式与非流式
# 精简化使用
# 用 onnxruntime 进行推理
# 1. 下载对应的模型
# 2. 加载模型
# 3. 端到端推理
# 4. 流式推理
import base64
import logging
import math

import numpy as np

from paddlespeech.server.engine.tts.online.onnx.tts_engine import TTSEngine
from paddlespeech.server.utils.audio_process import float2pcm
from paddlespeech.server.utils.config import get_config
from paddlespeech.server.utils.util import denorm
from paddlespeech.server.utils.util import get_chunks
from paddlespeech.t2s.frontend.zh_frontend import Frontend


class TTS:
    def __init__(self, config_path):
        self.config = get_config(config_path)['tts_online-onnx']
        self.config['voc_block'] = 36
        self.engine = TTSEngine()
        self.engine.init(self.config)
        self.executor = self.engine.executor
        #self.engine.warm_up()

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

        input_ids = self.frontend.get_input_ids(text,
                                                merge_sentences=merge_sentences,
                                                get_tone_ids=get_tone_ids)
        phone_ids = input_ids["phone_ids"]
        wav_list = []
        for i in range(len(phone_ids)):
            orig_hs = self.engine.executor.am_encoder_infer_sess.run(
                None, input_feed={'text': phone_ids[i].numpy()})
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
            mel = denorm(normalized_mel, self.engine.executor.am_mu,
                         self.engine.executor.am_std)
            wav = self.engine.executor.voc_sess.run(output_names=None,
                                                    input_feed={'logmel':
                                                                mel})[0]
            wav_list.append(wav)
        wavs = np.concatenate(wav_list)
        return wavs

    def streamTTS(self, text):

        get_tone_ids = False
        merge_sentences = False

        # front
        input_ids = self.frontend.get_input_ids(text,
                                                merge_sentences=merge_sentences,
                                                get_tone_ids=get_tone_ids)
        phone_ids = input_ids["phone_ids"]

        for i in range(len(phone_ids)):
            part_phone_ids = phone_ids[i].numpy()
            voc_chunk_id = 0

            # fastspeech2_csmsc
            if self.config.am == "fastspeech2_csmsc_onnx":
                # am
                mel = self.executor.am_sess.run(
                    output_names=None, input_feed={'text': part_phone_ids})
                mel = mel[0]

                # voc streaming
                mel_chunks = get_chunks(mel, self.config.voc_block,
                                        self.config.voc_pad, "voc")
                voc_chunk_num = len(mel_chunks)
                for i, mel_chunk in enumerate(mel_chunks):
                    sub_wav = self.executor.voc_sess.run(
                        output_names=None, input_feed={'logmel': mel_chunk})
                    sub_wav = self.depadding(sub_wav[0], voc_chunk_num, i,
                                             self.config.voc_block,
                                             self.config.voc_pad,
                                             self.config.voc_upsample)

                    yield self.after_process(sub_wav)

            # fastspeech2_cnndecoder_csmsc
            elif self.config.am == "fastspeech2_cnndecoder_csmsc_onnx":
                # am
                orig_hs = self.executor.am_encoder_infer_sess.run(
                    None, input_feed={'text': part_phone_ids})
                orig_hs = orig_hs[0]

                # streaming voc chunk info
                mel_len = orig_hs.shape[1]
                voc_chunk_num = math.ceil(mel_len / self.config.voc_block)
                start = 0
                end = min(self.config.voc_block + self.config.voc_pad, mel_len)

                # streaming am
                hss = get_chunks(orig_hs, self.config.am_block,
                                 self.config.am_pad, "am")
                am_chunk_num = len(hss)
                for i, hs in enumerate(hss):
                    am_decoder_output = self.executor.am_decoder_sess.run(
                        None, input_feed={'xs': hs})
                    am_postnet_output = self.executor.am_postnet_sess.run(
                        None,
                        input_feed={
                            'xs': np.transpose(am_decoder_output[0], (0, 2, 1))
                        })
                    am_output_data = am_decoder_output + np.transpose(
                        am_postnet_output[0], (0, 2, 1))
                    normalized_mel = am_output_data[0][0]

                    sub_mel = denorm(normalized_mel, self.executor.am_mu,
                                     self.executor.am_std)
                    sub_mel = self.depadding(sub_mel, am_chunk_num, i,
                                             self.config.am_block,
                                             self.config.am_pad, 1)

                    if i == 0:
                        mel_streaming = sub_mel
                    else:
                        mel_streaming = np.concatenate((mel_streaming, sub_mel),
                                                       axis=0)

                    # streaming voc
                    # 当流式AM推理的mel帧数大于流式voc推理的chunk size，开始进行流式voc 推理
                    while (mel_streaming.shape[0] >= end
                           and voc_chunk_id < voc_chunk_num):
                        voc_chunk = mel_streaming[start:end, :]

                        sub_wav = self.executor.voc_sess.run(
                            output_names=None, input_feed={'logmel': voc_chunk})
                        sub_wav = self.depadding(sub_wav[0], voc_chunk_num,
                                                 voc_chunk_id,
                                                 self.config.voc_block,
                                                 self.config.voc_pad,
                                                 self.config.voc_upsample)

                        yield self.after_process(sub_wav)

                        voc_chunk_id += 1
                        start = max(
                            0, voc_chunk_id * self.config.voc_block -
                            self.config.voc_pad)
                        end = min((voc_chunk_id + 1) * self.config.voc_block +
                                  self.config.voc_pad, mel_len)

            else:
                logging.error(
                    "Only support fastspeech2_csmsc or fastspeech2_cnndecoder_csmsc on streaming tts."
                )

    def streamTTSBytes(self, text):
        for wav in self.engine.executor.infer(text=text,
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
