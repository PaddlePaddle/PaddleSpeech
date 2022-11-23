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
"""Socket client to send wav to ASR server."""
import argparse
import wave

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

WAVE_OUTPUT_FILENAME = "output.wav"


def main():
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'rb')
    nframe = wf.getnframes()
    data = wf.readframes(nframe)
    print(f"Wave: {WAVE_OUTPUT_FILENAME}")
    print(f"Wave samples: {nframe}")
    print(f"Wave channels: {wf.getnchannels()}")
    print(f"Wave sample rate: {wf.getframerate()}")
    print(f"Wave sample width: {wf.getsampwidth()}")
    assert isinstance(data, bytes)
    socket_send(args.host_ip, args.host_port, data)


if __name__ == "__main__":
    main()
