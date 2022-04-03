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


def get_chunks(seg_dur, audio_id, audio_duration):
    """Get all chunk segments from a utterance

    Args:
        seg_dur (float): segment chunk duration, seconds
        audio_id (str): utterance name, 
        audio_duration (float): utterance duration, seconds

    Returns:
        List: all the chunk segments 
    """
    num_chunks = int(audio_duration / seg_dur)  # all in seconds
    chunk_lst = [
        audio_id + "_" + str(i * seg_dur) + "_" + str(i * seg_dur + seg_dur)
        for i in range(num_chunks)
    ]
    return chunk_lst
