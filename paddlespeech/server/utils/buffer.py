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
                 frame_duration_ms=80,
                 shift_ms=40,
                 sample_rate=16000,
                 sample_width=2):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.shift_ms = shift_ms
        self.remained_audio = b''
        self.sample_width = sample_width  # int16 = 2; float32 = 4

    def frame_generator(self, audio):
        """Generates audio frames from PCM audio data.
        Takes the desired frame duration in milliseconds, the PCM data, and
        the sample rate.
        Yields Frames of the requested duration.
        """
        audio = self.remained_audio + audio
        self.remained_audio = b''

        n = int(self.sample_rate * (self.frame_duration_ms / 1000.0) *
                self.sample_width)
        shift_n = int(self.sample_rate * (self.shift_ms / 1000.0) *
                      self.sample_width)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / self.sample_rate) / self.sample_width
        shift_duration = (float(shift_n) / self.sample_rate) / self.sample_width
        while offset + n <= len(audio):
            yield Frame(audio[offset:offset + n], timestamp, duration)
            timestamp += shift_duration
            offset += shift_n

        self.remained_audio += audio[offset:]
