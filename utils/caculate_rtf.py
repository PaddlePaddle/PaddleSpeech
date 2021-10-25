#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import argparse
import codecs
import glob
import os

from dateutil import parser


def get_parser():
    parser = argparse.ArgumentParser(
        description="calculate real time factor (RTF)")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="path to logging directory", )
    return parser


def main():

    args = get_parser().parse_args()

    audio_sec = 0
    decode_sec = 0
    n_utt = 0

    audio_durations = []
    start_times = []
    end_times = []
    for x in glob.glob(os.path.join(args.log_dir, "decode.*.log")):
        with codecs.open(x, "r", "utf-8") as f:
            for line in f:
                x = line.strip()
                # 2021-10-25 08:22:04.052 | INFO | xxx:recog_v2:188 - feat: (1570, 83)
                if "feat:" in x:
                    dur = int(x.split("(")[1].split(',')[0])
                    audio_durations += [dur]
                    start_times += [parser.parse(x.split("|")[0])]
                elif "total log probability:" in x:
                    end_times += [parser.parse(x.split("|")[0])]
        assert len(audio_durations) == len(end_times), (len(audio_durations),
                                                        len(end_times), )
        assert len(start_times) == len(end_times), (len(start_times),
                                                    len(end_times))

        audio_sec += sum(audio_durations) / 100  # [sec]
        decode_sec += sum([(end - start).total_seconds()
                           for start, end in zip(start_times, end_times)])
        n_utt += len(audio_durations)

    print("Total audio duration: %.3f [sec]" % audio_sec)
    print("Total decoding time: %.3f [sec]" % decode_sec)
    rtf = decode_sec / audio_sec if audio_sec > 0 else 0
    print("RTF: %.3f" % rtf)
    latency = decode_sec * 1000 / n_utt if n_utt > 0 else 0
    print("Latency: %.3f [ms/sentence]" % latency)


if __name__ == "__main__":
    main()
