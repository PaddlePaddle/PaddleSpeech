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


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


class ChunkBuffer(object):
    def __init__(self,
                 window_n=7,
                 shift_n=4,
                 window_ms=20,
                 shift_ms=10,
                 sample_rate=16000,
                 sample_width=2):
        """audio sample data point buffer

        Args:
            window_n (int, optional): decode window frame length. Defaults to 7 frame.
            shift_n (int, optional): decode shift frame length. Defaults to 4 frame.
            window_ms (int, optional): frame length, ms. Defaults to 20 ms.
            shift_ms (int, optional): shift length, ms. Defaults to 10 ms.
            sample_rate (int, optional): audio sample rate. Defaults to 16000.
            sample_width (int, optional): sample point bytes. Defaults to 2 bytes.
        """
        self.window_n = window_n
        self.shift_n = shift_n
        self.window_ms = window_ms
        self.shift_ms = shift_ms
        self.sample_rate = sample_rate
        self.sample_width = sample_width  # int16 = 2; float32 = 4

        self.window_sec = float((self.window_n - 1) * self.shift_ms +
                                self.window_ms) / 1000.0
        self.shift_sec = float(self.shift_n * self.shift_ms / 1000.0)

        self.window_bytes = int(self.window_sec * self.sample_rate *
                                self.sample_width)
        self.shift_bytes = int(self.shift_sec * self.sample_rate *
                               self.sample_width)

        self.remained_audio = b''
        # abs timestamp from `start` or latest `reset`
        self.timestamp = 0.0

    def reset(self):
        """
            reset buffer state.
        """
        self.timestamp = 0.0
        self.remained_audio = b''

    def frame_generator(self, audio):
        """Generates audio frames from PCM audio data.
        Takes the desired frame duration in milliseconds, the PCM data, and
        the sample rate.
        Yields Frames of the requested duration.
        """
        audio = self.remained_audio + audio
        self.remained_audio = b''

        offset = 0
        while offset + self.window_bytes <= len(audio):
            yield Frame(audio[offset:offset + self.window_bytes],
                        self.timestamp, self.window_sec)
            self.timestamp += self.shift_sec
            offset += self.shift_bytes

        self.remained_audio += audio[offset:]
