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
import argparse
import asyncio
import os

from paddlespeech.server.utils.util import compute_delay
from paddlespeech.t2s.exps.syn_utils import get_sentences


def test(args, text, utt_id):
    output = str(args.output_dir + "/" + utt_id + ".wav")
    if args.protocol == "http":
        print("tts http client start")
        from paddlespeech.server.utils.audio_handler import TTSHttpHandler
        handler = TTSHttpHandler(args.server_ip, args.port, args.play)
        first_response, final_response, duration, save_audio_success, receive_time_list, chunk_duration_list = handler.run(
            text, args.spk_id, args.speed, args.volume, args.sample_rate,
            output)

    elif args.protocol == "websocket":
        from paddlespeech.server.utils.audio_handler import TTSWsHandler
        print("tts websocket client start")
        handler = TTSWsHandler(args.server_ip, args.port, args.play)
        loop = asyncio.get_event_loop()
        first_response, final_response, duration, save_audio_success, receive_time_list, chunk_duration_list = loop.run_until_complete(
            handler.run(text, output))

    else:
        print("Please set correct protocol, http or websocket")

    return first_response, final_response, duration, save_audio_success, receive_time_list, chunk_duration_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        default="../../../../../../paddlespeech/t2s/exps/csmsc_test.txt",
        help="text to synthesize, a 'utt_id sentence' pair per line")
    parser.add_argument('--spk_id', type=int, default=0, help='Speaker id')
    parser.add_argument('--speed', type=float, default=1.0, help='Audio speed')
    parser.add_argument('--volume',
                        type=float,
                        default=1.0,
                        help='Audio volume')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=0,
        help='Sampling rate, the default is the same as the model')
    parser.add_argument("--server_ip",
                        type=str,
                        help="server ip",
                        default="127.0.0.1")
    parser.add_argument("--port", type=int, help="server port", default=8092)
    parser.add_argument("--protocol",
                        type=str,
                        choices=['http', 'websocket'],
                        help="server protocol",
                        default="http")
    parser.add_argument("--output_dir",
                        type=str,
                        default="./output",
                        help="output dir")
    parser.add_argument("--play",
                        type=bool,
                        help="whether to play audio",
                        default=False)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    first_response_list = []
    final_response_list = []
    duration_list = []
    all_delay_list = []
    packet_count = 0.0

    sentences = get_sentences(text_file=args.text, lang="zh")
    for utt_id, sentence in sentences:
        first_response, final_response, duration, save_audio_success, receive_time_list, chunk_duration_list = test(
            args, sentence, utt_id)
        delay_time_list = compute_delay(receive_time_list, chunk_duration_list)
        first_response_list.append(first_response)
        final_response_list.append(final_response)
        duration_list.append(duration)
        packet_count += len(receive_time_list)

        print(f"句子：{sentence}")
        print(f"首包响应时间：{first_response} s")
        print(f"尾包响应时间：{final_response} s")
        print(f"音频时长：{duration} s")
        print(f"该句RTF：{final_response/duration}")

        if delay_time_list != []:
            for t in delay_time_list:
                all_delay_list.append(t)
            print(
                f"该句流式合成的延迟情况：总包个数：{len(receive_time_list)}，延迟包个数：{len(delay_time_list)}, 最小延迟时间：{min(delay_time_list)} s, 最大延迟时间：{max(delay_time_list)} s, 平均延迟时间：{sum(delay_time_list)/len(delay_time_list)} s, 延迟率：{len(delay_time_list)/len(receive_time_list)}"
            )
        else:
            print("该句流式合成无延迟情况")

        packet_count += len(receive_time_list)

    assert (len(first_response_list) == len(final_response_list)
            and len(final_response_list) == len(duration_list))

    avg_first_response = sum(first_response_list) / len(first_response_list)
    avg_final_response = sum(final_response_list) / len(final_response_list)
    avg_duration = sum(duration_list) / len(duration_list)
    RTF = sum(final_response_list) / sum(duration_list)
    if all_delay_list != []:
        delay_count = len(all_delay_list)
        avg_delay = sum(all_delay_list) / len(all_delay_list)
        delay_ratio = len(all_delay_list) / packet_count
        min_delay = min(all_delay_list)
        max_delay = max(all_delay_list)
    else:
        delay_count = 0.0
        avg_delay = 0.0
        delay_ratio = 0.0
        min_delay = 0.0
        max_delay = 0.0

    print(
        "************************* server/client result ***************************************"
    )
    print(
        f"test num: {len(duration_list)}, avg first response: {avg_first_response} s, avg final response: {avg_final_response} s, avg duration: {avg_duration}, RTF: {RTF}."
    )
    print(
        f"test num: {len(duration_list)}, packet count: {packet_count}, delay count: {delay_count}, avg delay time: {avg_delay} s, delay ratio: {delay_ratio} "
    )
    print(
        f"min duration: {min(duration_list)} s, max duration: {max(duration_list)} s"
    )
    print(
        f"min first response: {min(first_response_list)} s, max first response: {max(first_response_list)} s."
    )
    print(
        f"min final response: {min(final_response_list)} s, max final response: {max(final_response_list)} s."
    )
    print(f"min delay: {min_delay} s, max delay: {max_delay}")
