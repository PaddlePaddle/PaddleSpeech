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
                 window_n=7,    # frame
                 shift_n=4,     # frame
                 window_ms=20,  # ms
                 shift_ms=10,   # ms
                 sample_rate=16000,
                 sample_width=2):
        self.window_n = window_n
        self.shift_n = shift_n
        self.window_ms = window_ms
        self.shift_ms = shift_ms
        self.sample_rate = sample_rate
        self.sample_width = sample_width  # int16 = 2; float32 = 4
        self.remained_audio = b''

        self.window_sec = float((self.window_n - 1) * self.shift_ms + self.window_ms) / 1000.0 
        self.shift_sec = float(self.shift_n * self.shift_ms / 1000.0)

        self.window_bytes = int(self.window_sec * self.sample_rate * self.sample_width)
        self.shift_bytes = int(self.shift_sec * self.sample_rate * self.sample_width)

    def frame_generator(self, audio):
        """Generates audio frames from PCM audio data.
        Takes the desired frame duration in milliseconds, the PCM data, and
        the sample rate.
        Yields Frames of the requested duration.
        """
        audio = self.remained_audio + audio
        self.remained_audio = b''

        offset = 0
        timestamp = 0.0

        while offset + self.window_bytes <= len(audio):
            yield Frame(audio[offset:offset + self.window_bytes], timestamp, self.window_sec)
            timestamp += self.shift_sec
            offset += self.shift_bytes

        self.remained_audio += audio[offset:]
