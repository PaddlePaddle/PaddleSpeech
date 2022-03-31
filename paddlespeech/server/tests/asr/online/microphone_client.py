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
import asyncio
import json
import logging
import threading
import wave
from signal import SIGINT
from signal import SIGTERM

import pyaudio
import websockets


class ASRAudioHandler(threading.Thread):
    def __init__(self, url="127.0.0.1", port=8091):
        threading.Thread.__init__(self)
        self.url = url
        self.port = port
        self.url = "ws://" + self.url + ":" + str(self.port) + "/ws/asr"
        self.fileName = "./output.wav"
        self.chunk = 5120
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self._running = True
        self._frames = []
        self.data_backup = []

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
        stream = p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk)
        while (self._running):
            data = stream.read(self.chunk)
            self._frames.append(data)
            self.data_backup.append(data)

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
        wf.writeframes(b''.join(self.data_backup))
        wf.close()
        p.terminate()

    def stoprecord(self):
        """
        stop recording
        """
        self._running = False

    async def run(self):
        aa = input("是否开始录音？   (y/n)")
        if aa.strip() == "y":
            self.startrecord()
            logging.info("*" * 10 + "开始录音，请输入语音")

            async with websockets.connect(self.url) as ws:
                # 发送开始指令
                audio_info = json.dumps(
                    {
                        "name": "test.wav",
                        "signal": "start",
                        "nbest": 5
                    },
                    sort_keys=True,
                    indent=4,
                    separators=(',', ': '))
                await ws.send(audio_info)
                msg = await ws.recv()
                logging.info("receive msg={}".format(msg))

                # send bytes data
                logging.info("结束录音请: Ctrl + c。继续请按回车。")
                try:
                    while True:
                        while len(self._frames) > 0:
                            await ws.send(self._frames.pop(0))
                            msg = await ws.recv()
                            logging.info("receive msg={}".format(msg))
                except asyncio.CancelledError:
                    # quit
                    # send finished 
                    audio_info = json.dumps(
                        {
                            "name": "test.wav",
                            "signal": "end",
                            "nbest": 5
                        },
                        sort_keys=True,
                        indent=4,
                        separators=(',', ': '))
                    await ws.send(audio_info)
                    msg = await ws.recv()
                    logging.info("receive msg={}".format(msg))

                    self.stoprecord()
                    logging.info("*" * 10 + "录音结束")
                    self.save()
        elif aa.strip() == "n":
            exit()
        else:
            print("无效输入!")
            exit()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.info("asr websocket client start")

    handler = ASRAudioHandler("127.0.0.1", 8091)
    loop = asyncio.get_event_loop()
    main_task = asyncio.ensure_future(handler.run())
    for signal in [SIGINT, SIGTERM]:
        loop.add_signal_handler(signal, main_task.cancel)
    try:
        loop.run_until_complete(main_task)
    finally:
        loop.close()

    logging.info("asr websocket client finished")
