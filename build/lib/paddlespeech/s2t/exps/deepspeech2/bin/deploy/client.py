# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""Client-end for the ASR demo."""
import argparse
import sys

import keyboard
import pyaudio

from paddlespeech.s2t.utils.socket_server import socket_send

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--host_ip",
    default="localhost",
    type=str,
    help="Server IP address. (default: %(default)s)")
parser.add_argument(
    "--host_port",
    default=8086,
    type=int,
    help="Server Port. (default: %(default)s)")
args = parser.parse_args()

is_recording = False
enable_trigger_record = True


def on_press_release(x):
    """Keyboard callback function."""
    global is_recording, enable_trigger_record
    press = keyboard.KeyboardEvent('down', 28, 'space')
    release = keyboard.KeyboardEvent('up', 28, 'space')
    if x.event_type == 'down' and x.name == press.name:
        if (not is_recording) and enable_trigger_record:
            sys.stdout.write("Start Recording ... ")
            sys.stdout.flush()
            is_recording = True
    if x.event_type == 'up' and x.name == release.name:
        if is_recording:
            is_recording = False


data_list = []


def callback(in_data, frame_count, time_info, status):
    """Audio recorder's stream callback function."""
    global data_list, is_recording, enable_trigger_record
    if is_recording:
        data_list.append(in_data)
        enable_trigger_record = False
    elif len(data_list) > 0:
        socket_send(args.host_ip, args.host_port, ''.join(data_list))
        data_list = []
    enable_trigger_record = True
    return (in_data, pyaudio.paContinue)


def main():
    # prepare audio recorder
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        stream_callback=callback)
    stream.start_stream()

    # prepare keyboard listener
    while (1):
        keyboard.hook(on_press_release)
        if keyboard.record('esc'):
            break

    # close up
    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    main()
