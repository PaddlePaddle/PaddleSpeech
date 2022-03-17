# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
record wave from the mic
"""

import threading
import pyaudio
import wave
import logging


class RecordThread(threading.Thread):
    """
    thread for wave record
    """
    def __init__(self):
        threading.Thread.__init__(self)
        self.fileName = "./input.wav"
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self._running = True
        self._frames = []

    def startrecord(self):
        """
        start a new thread to record wave
        """
        threading._start_new_thread(self.recording, ())

    def recording(self):
        """
        recording wave
        """
        self._running = True
        self._frames = []
        p = pyaudio.PyAudio()
        stream = p.open(format=self.format,
                               channels=self.channels,
                               rate=self.rate,
                               input=True,
                               frames_per_buffer=self.chunk)
        while(self._running):
            data = stream.read(self.chunk)
            self._frames.append(data)
 
        stream.stop_stream()
        stream.close()
        p.terminate()

    def save(self):
        """
        save wave data
        """
        p = pyaudio.PyAudio()
        wf = wave.open(self.fileName, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self._frames))
        wf.close()
        p.terminate()

    def stoprecord(self):
        """
        stop recording
        """
        self._running = False

    def get_audio(self, filepath, sample_rate):
        """
        record wave and save result
        """
        self.fileName = filepath
        self.rate = sample_rate
        aa = input("是否开始录音？   (y/n)")
        if aa.strip() == "y":
            self.startrecord()
            logging.info("*" * 10 + "开始录音，请输入语音")
            while True:
                bb = input("是否结束录音？   (y/n)")
                if bb.strip() == "y":
                    self.stoprecord()
                    logging.info("*" * 10 + "录音结束")
                    self.save()
                    break
        elif aa.strip() == "n":
            exit()
        else:
            print("无效输入!")
            exit()

if __name__ == "__main__":
    input_filename = "temp.wav"        # 麦克风采集的语音输入
    input_filepath = "./"              # 输入文件的path
    in_path = input_filepath + input_filename

    audio_record = RecordThread()
    audio_record.get_audio(in_path, 16000)
