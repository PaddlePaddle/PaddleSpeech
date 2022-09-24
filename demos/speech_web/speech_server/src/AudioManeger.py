import datetime
import imp
import os
import random
import wave
from queue import Queue

import numpy as np

from .util import randName


class AudioMannger:
    def __init__(self,
                 robot,
                 frame_length=160,
                 frame=10,
                 data_width=2,
                 vad_default=300):
        # 二进制 pcm 流 
        self.audios = b''
        self.asr_result = ""
        # Speech 核心主体
        self.robot = robot

        self.file_dir = "source"
        os.makedirs(self.file_dir, exist_ok=True)
        self.vad_deafult = vad_default
        self.vad_threshold = vad_default
        self.vad_threshold_path = os.path.join(self.file_dir,
                                               "vad_threshold.npy")

        # 10ms 一帧
        self.frame_length = frame_length
        # 10帧，检测一次 vad
        self.frame = frame
        # int 16, 两个bytes
        self.data_width = data_width
        # window
        self.window_length = frame_length * frame * data_width

        # 是否开始录音
        self.on_asr = False
        self.silence_cnt = 0
        self.max_silence_cnt = 4
        self.is_pause = False  # 录音暂停与恢复

    def init(self):
        if os.path.exists(self.vad_threshold_path):
            # 平均响度文件存在
            self.vad_threshold = np.load(self.vad_threshold_path)

    def clear_audio(self):
        # 清空 pcm 累积片段与 asr 识别结果
        self.audios = b''

    def clear_asr(self):
        self.asr_result = ""

    def compute_chunk_volume(self, start_index, pcm_bins):
        # 根据帧长计算能量平均值
        pcm_bin = pcm_bins[start_index:start_index + self.window_length]
        # 转成 numpy
        pcm_np = np.frombuffer(pcm_bin, np.int16)
        # 归一化 + 计算响度
        x = pcm_np.astype(np.float32)
        x = np.abs(x)
        return np.mean(x)

    def is_speech(self, start_index, pcm_bins):
        # 检查是否没
        if start_index > len(pcm_bins):
            return False
        # 检查从这个 start 开始是否为静音帧
        energy = self.compute_chunk_volume(
            start_index=start_index, pcm_bins=pcm_bins)
        # print(energy)
        if energy > self.vad_threshold:
            return True
        else:
            return False

    def compute_env_volume(self, pcm_bins):
        max_energy = 0
        start = 0
        while start < len(pcm_bins):
            energy = self.compute_chunk_volume(
                start_index=start, pcm_bins=pcm_bins)
            if energy > max_energy:
                max_energy = energy
            start += self.window_length
        self.vad_threshold = max_energy + 100 if max_energy > self.vad_deafult else self.vad_deafult

        # 保存成文件
        np.save(self.vad_threshold_path, self.vad_threshold)
        print(f"vad 阈值大小: {self.vad_threshold}")
        print(f"环境采样保存: {os.path.realpath(self.vad_threshold_path)}")

    def stream_asr(self, pcm_bin):
        # 先把 pcm_bin 送进去做端点检测
        start = 0
        while start < len(pcm_bin):
            if self.is_speech(start_index=start, pcm_bins=pcm_bin):
                self.on_asr = True
                self.silence_cnt = 0
                print("录音中")
                self.audios += pcm_bin[start:start + self.window_length]
            else:
                if self.on_asr:
                    self.silence_cnt += 1
                    if self.silence_cnt > self.max_silence_cnt:
                        self.on_asr = False
                        self.silence_cnt = 0
                        # 录音停止
                        print("录音停止")
                        # audios 保存为 wav, 送入 ASR
                        if len(self.audios) > 2 * 16000:
                            file_path = os.path.join(
                                self.file_dir,
                                "asr_" + datetime.datetime.strftime(
                                    datetime.datetime.now(),
                                    '%Y%m%d%H%M%S') + randName() + ".wav")
                            self.save_audio(file_path=file_path)
                            self.asr_result = self.robot.speech2text(file_path)
                        self.clear_audio()
                        return self.asr_result
                    else:
                        # 正常接收
                        print("录音中 静音")
                        self.audios += pcm_bin[start:start + self.window_length]
            start += self.window_length
        return ""

    def save_audio(self, file_path):
        print("保存音频")
        wf = wave.open(file_path, 'wb')  # 创建一个音频文件，名字为“01.wav"
        wf.setnchannels(1)  # 设置声道数为2
        wf.setsampwidth(2)  # 设置采样深度为
        wf.setframerate(16000)  # 设置采样率为16000
        # 将数据写入创建的音频文件
        wf.writeframes(self.audios)
        # 写完后将文件关闭
        wf.close()

    def end(self):
        # audios 保存为 wav, 送入 ASR
        file_path = os.path.join(self.file_dir, "asr.wav")
        self.save_audio(file_path=file_path)
        return self.robot.speech2text(file_path)

    def stop(self):
        self.is_pause = True
        self.audios = b''

    def resume(self):
        self.is_pause = False
