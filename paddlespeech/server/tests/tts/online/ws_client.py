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
import argparse
import asyncio

from paddlespeech.server.utils.audio_handler import TTSWsHandler
from paddlespeech.server.utils.util import compute_delay

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        help="A sentence to be synthesized",
        default="您好，欢迎使用语音合成服务。")
    parser.add_argument(
        "--server", type=str, help="server ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, help="server port", default=8092)
    parser.add_argument(
        "--output", type=str, help="save audio path", default=None)
    parser.add_argument(
        "--play", type=bool, help="whether to play audio", default=False)
    args = parser.parse_args()

    print("tts websocket client start")
    handler = TTSWsHandler(args.server, args.port, args.play)
    loop = asyncio.get_event_loop()
    first_response, final_response, duration, save_audio_success, receive_time_list, chunk_duration_list = loop.run_until_complete(
        handler.run(args.text, args.output))
    delay_time_list = compute_delay(receive_time_list, chunk_duration_list)

    print(f"sentence: {args.text}")
    print(f"duration: {duration} s")
    print(f"first response: {first_response} s")
    print(f"final response: {final_response} s")
    print(f"RTF: {final_response/duration}")
    if args.output is not None:
        if save_audio_success:
            print(f"Audio successfully saved in {args.output}")
        else:
            print("Audio save failed.")

    if delay_time_list != []:
        print(
            f"Delay situation: total number of packages: {len(receive_time_list)}, the number of delayed packets: {len(delay_time_list)}, minimum delay time: {min(delay_time_list)} s, maximum delay time: {max(delay_time_list)} s, average delay time: {sum(delay_time_list)/len(delay_time_list)} s, delay rate:{len(delay_time_list)/len(receive_time_list)}"
        )
    else:
        print("The sentence has no delay in streaming synthesis.")
