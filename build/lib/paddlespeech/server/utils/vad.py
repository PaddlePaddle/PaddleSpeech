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
import collections

import webrtcvad


class VADAudio():
    def __init__(self,
                 aggressiveness=2,
                 rate=16000,
                 frame_duration_ms=20,
                 sample_width=2,
                 padding_ms=200,
                 padding_ratio=0.9):
        """Initializes VAD with given aggressivenes and sets up internal queues"""
        self.vad = webrtcvad.Vad(aggressiveness)
        self.rate = rate
        self.sample_width = sample_width
        self.frame_duration_ms = frame_duration_ms
        self._frame_length = int(rate * (frame_duration_ms / 1000.0) *
                                 self.sample_width)
        self._buffer_queue = collections.deque()
        self.ring_buffer = collections.deque(maxlen=padding_ms //
                                             frame_duration_ms)
        self._ratio = padding_ratio
        self.triggered = False

    def add_audio(self, audio):
        """Adds new audio to internal queue"""
        for x in audio:
            self._buffer_queue.append(x)

    def frame_generator(self):
        """Generator that yields audio frames of frame_duration_ms"""
        while len(self._buffer_queue) > self._frame_length:
            frame = bytearray()
            for _ in range(self._frame_length):
                frame.append(self._buffer_queue.popleft())
            yield bytes(frame)

    def vad_collector(self):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        for frame in self.frame_generator():
            is_speech = self.vad.is_speech(frame, self.rate)
            if not self.triggered:
                self.ring_buffer.append((frame, is_speech))
                num_voiced = len(
                    [f for f, speech in self.ring_buffer if speech])
                if num_voiced > self._ratio * self.ring_buffer.maxlen:
                    self.triggered = True
                    for f, s in self.ring_buffer:
                        yield f
                    self.ring_buffer.clear()
            else:
                yield frame
                self.ring_buffer.append((frame, is_speech))
                num_unvoiced = len(
                    [f for f, speech in self.ring_buffer if not speech])
                if num_unvoiced > self._ratio * self.ring_buffer.maxlen:
                    self.triggered = False
                    yield None
                    self.ring_buffer.clear()
