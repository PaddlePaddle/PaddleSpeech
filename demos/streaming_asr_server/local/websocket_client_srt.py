#!/usr/bin/python
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
# calc avg RTF(NOT Accurate): grep -rn RTF log.txt | awk '{print $NF}' | awk -F "=" '{sum += $NF} END {print "all time",sum, "audio num", NR,  "RTF", sum/NR}'
# python3 websocket_client.py --server_ip 127.0.0.1 --port 8290 --punc.server_ip 127.0.0.1 --punc.port 8190 --wavfile ./zh.wav
# python3 websocket_client.py --server_ip 127.0.0.1 --port 8290 --wavfile ./zh.wav
import argparse
import asyncio
import codecs
import os
from pydub import AudioSegment
import re

from paddlespeech.cli.log import logger
from paddlespeech.server.utils.audio_handler import ASRWsAudioHandler

def convert_to_wav(input_file):
    # Load audio file
    audio = AudioSegment.from_file(input_file)

    # Set parameters for audio file
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)

    # Create output filename
    output_file = os.path.splitext(input_file)[0] + ".wav"

    # Export audio file as WAV
    audio.export(output_file, format="wav")

    logger.info(f"{input_file} converted to {output_file}")

def format_time(sec):
    # Convert seconds to SRT format (HH:MM:SS,ms)
    hours = int(sec/3600)
    minutes = int((sec%3600)/60)
    seconds = int(sec%60)
    milliseconds = int((sec%1)*1000)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}'

def results2srt(results, srt_file):
    """convert results from paddlespeech to srt format for subtitle
    Args:
        results (dict): results from paddlespeech
    """
    # times contains start and end time of each word
    times = results['times']
    # result contains the whole sentence including punctuation
    result = results['result']
    # split result into several sencences by '，' and '。'
    sentences = re.split('，|。', result)[:-1]
    # print("sentences: ", sentences)
    # generate relative time for each sentence in sentences
    relative_times = []
    word_i = 0
    for sentence in sentences:
        relative_times.append([])
        for word in sentence:
            if relative_times[-1] == []:
                relative_times[-1].append(times[word_i]['bg'])
            if len(relative_times[-1]) == 1:
                relative_times[-1].append(times[word_i]['ed'])
            else:
                relative_times[-1][1] = times[word_i]['ed']
            word_i += 1
    # print("relative_times: ", relative_times)
    # generate srt file acoording to relative_times and sentences
    with open(srt_file, 'w') as f:
        for i in range(len(sentences)):
            # Write index number
            f.write(str(i+1)+'\n')
            
            # Write start and end times
            start = format_time(relative_times[i][0])
            end = format_time(relative_times[i][1])
            f.write(start + ' --> ' + end + '\n')
            
            # Write text
            f.write(sentences[i]+'\n\n')
    logger.info(f"results saved to {srt_file}")

def main(args):
    logger.info("asr websocket client start")
    handler = ASRWsAudioHandler(
        args.server_ip,
        args.port,
        endpoint=args.endpoint,
        punc_server_ip=args.punc_server_ip,
        punc_server_port=args.punc_server_port)
    loop = asyncio.get_event_loop()

    # check if the wav file is mp3 format
    # if so, convert it to wav format using convert_to_wav function
    if args.wavfile and os.path.exists(args.wavfile):
        if args.wavfile.endswith(".mp3"):
            convert_to_wav(args.wavfile)
            args.wavfile = args.wavfile.replace(".mp3", ".wav")

    # support to process single audio file
    if args.wavfile and os.path.exists(args.wavfile):
        logger.info(f"start to process the wavscp: {args.wavfile}")
        result = loop.run_until_complete(handler.run(args.wavfile))
        # result = result["result"]
        # logger.info(f"asr websocket client finished : {result}")
        results2srt(result, args.wavfile.replace(".wav", ".srt"))

    # support to process batch audios from wav.scp
    if args.wavscp and os.path.exists(args.wavscp):
        logger.info(f"start to process the wavscp: {args.wavscp}")
        with codecs.open(args.wavscp, 'r', encoding='utf-8') as f,\
             codecs.open("result.txt", 'w', encoding='utf-8') as w:
            for line in f:
                utt_name, utt_path = line.strip().split()
                result = loop.run_until_complete(handler.run(utt_path))
                result = result["result"]
                w.write(f"{utt_name} {result}\n")


if __name__ == "__main__":
    logger.info("Start to do streaming asr client")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--server_ip', type=str, default='127.0.0.1', help='server ip')
    parser.add_argument('--port', type=int, default=8090, help='server port')
    parser.add_argument(
        '--punc.server_ip',
        type=str,
        default=None,
        dest="punc_server_ip",
        help='Punctuation server ip')
    parser.add_argument(
        '--punc.port',
        type=int,
        default=8091,
        dest="punc_server_port",
        help='Punctuation server port')
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/paddlespeech/asr/streaming",
        help="ASR websocket endpoint")
    parser.add_argument(
        "--wavfile",
        action="store",
        help="wav file path ",
        default="./16_audio.wav")
    parser.add_argument(
        "--wavscp", type=str, default=None, help="The batch audios dict text")
    args = parser.parse_args()

    main(args)
