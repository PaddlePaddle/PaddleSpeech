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
import logging
import math
import threading
import time
from pathlib import Path

import numpy as np
import paddle
import soundfile as sf
import yaml
from yacs.config import CfgNode

from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.t2s.exps.syn_utils import get_am_inference
from paddlespeech.t2s.exps.syn_utils import get_frontend
from paddlespeech.t2s.exps.syn_utils import get_sentences
from paddlespeech.t2s.exps.syn_utils import get_voc_inference
from paddlespeech.t2s.exps.syn_utils import model_alias
from paddlespeech.t2s.utils import str2bool

mel_streaming = None
wav_streaming = None
streaming_first_time = 0.0
streaming_voc_st = 0.0
sample_rate = 0


def denorm(data, mean, std):
    return data * std + mean


def get_chunks(data, block_size, pad_size, step):
    if step == "am":
        data_len = data.shape[1]
    elif step == "voc":
        data_len = data.shape[0]
    else:
        print("Please set correct type to get chunks, am or voc")

    chunks = []
    n = math.ceil(data_len / block_size)
    for i in range(n):
        start = max(0, i * block_size - pad_size)
        end = min((i + 1) * block_size + pad_size, data_len)
        if step == "am":
            chunks.append(data[:, start:end, :])
        elif step == "voc":
            chunks.append(data[start:end, :])
        else:
            print("Please set correct type to get chunks, am or voc")
    return chunks


def get_streaming_am_inference(args, am_config):
    with open(args.phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    print("vocab_size:", vocab_size)

    am_name = "fastspeech2"
    odim = am_config.n_mels

    am_class = dynamic_import(am_name, model_alias)
    am = am_class(idim=vocab_size, odim=odim, **am_config["model"])
    am.set_state_dict(paddle.load(args.am_ckpt)["main_params"])
    am.eval()
    am_mu, am_std = np.load(args.am_stat)
    am_mu = paddle.to_tensor(am_mu)
    am_std = paddle.to_tensor(am_std)

    return am, am_mu, am_std


def init(args):
    global sample_rate
    # get config
    with open(args.am_config) as f:
        am_config = CfgNode(yaml.safe_load(f))
    with open(args.voc_config) as f:
        voc_config = CfgNode(yaml.safe_load(f))

    sample_rate = am_config.fs

    # frontend
    frontend = get_frontend(args)

    # acoustic model
    if args.am == 'fastspeech2_cnndecoder_csmsc':
        am, am_mu, am_std = get_streaming_am_inference(args, am_config)
        am_infer_info = [am, am_mu, am_std, am_config]
    else:
        am_inference, am_name, am_dataset = get_am_inference(args, am_config)
        am_infer_info = [am_inference, am_name, am_dataset, am_config]

    # vocoder
    voc_inference = get_voc_inference(args, voc_config)
    voc_infer_info = [voc_inference, voc_config]

    return frontend, am_infer_info, voc_infer_info


def get_phone(args, frontend, sentence, merge_sentences, get_tone_ids):
    am_name = args.am[:args.am.rindex('_')]
    tone_ids = None

    if args.lang == 'zh':
        input_ids = frontend.get_input_ids(
            sentence,
            merge_sentences=merge_sentences,
            get_tone_ids=get_tone_ids)
        phone_ids = input_ids["phone_ids"]
        if get_tone_ids:
            tone_ids = input_ids["tone_ids"]
    elif args.lang == 'en':
        input_ids = frontend.get_input_ids(
            sentence, merge_sentences=merge_sentences)
        phone_ids = input_ids["phone_ids"]
    else:
        print("lang should in {'zh', 'en'}!")

    return phone_ids, tone_ids


@paddle.no_grad()
# 生成完整的mel
def gen_mel(args, am_infer_info, part_phone_ids, part_tone_ids):
    # 如果是支持流式的AM模型
    if args.am == 'fastspeech2_cnndecoder_csmsc':
        am, am_mu, am_std, am_config = am_infer_info
        orig_hs, h_masks = am.encoder_infer(part_phone_ids)
        if args.am_streaming:
            am_pad = args.am_pad
            am_block = args.am_block
            hss = get_chunks(orig_hs, am_block, am_pad, "am")
            chunk_num = len(hss)
            mel_list = []
            for i, hs in enumerate(hss):
                before_outs, _ = am.decoder(hs)
                after_outs = before_outs + am.postnet(
                    before_outs.transpose((0, 2, 1))).transpose((0, 2, 1))
                normalized_mel = after_outs[0]
                sub_mel = denorm(normalized_mel, am_mu, am_std)
                # clip output part of pad
                if i == 0:
                    sub_mel = sub_mel[:-am_pad]
                elif i == chunk_num - 1:
                    # 最后一块的右侧一定没有 pad 够
                    sub_mel = sub_mel[am_pad:]
                else:
                    # 倒数几块的右侧也可能没有 pad 够
                    sub_mel = sub_mel[am_pad:(am_block + am_pad) -
                                      sub_mel.shape[0]]
                mel_list.append(sub_mel)
                mel = paddle.concat(mel_list, axis=0)

        else:
            orig_hs, h_masks = am.encoder_infer(part_phone_ids)
            before_outs, _ = am.decoder(orig_hs)
            after_outs = before_outs + am.postnet(
                before_outs.transpose((0, 2, 1))).transpose((0, 2, 1))
            normalized_mel = after_outs[0]
            mel = denorm(normalized_mel, am_mu, am_std)

    else:
        am_inference, am_name, am_dataset, am_config = am_infer_info
        mel = am_inference(part_phone_ids)

    return mel


@paddle.no_grad()
def streaming_voc_infer(args, voc_infer_info, mel_len):
    global mel_streaming
    global streaming_first_time
    global wav_streaming
    voc_inference, voc_config = voc_infer_info
    block = args.voc_block
    pad = args.voc_pad
    upsample = voc_config.n_shift
    wav_list = []
    flag = 1

    valid_start = 0
    valid_end = min(valid_start + block, mel_len)
    actual_start = 0
    actual_end = min(valid_end + pad, mel_len)
    mel_chunk = mel_streaming[actual_start:actual_end, :]

    while valid_end <= mel_len:
        sub_wav = voc_inference(mel_chunk)
        if flag == 1:
            streaming_first_time = time.time()
            flag = 0

        # get valid wav    
        start = valid_start - actual_start
        if valid_end == mel_len:
            sub_wav = sub_wav[start * upsample:]
            wav_list.append(sub_wav)
            break
        else:
            end = start + block
            sub_wav = sub_wav[start * upsample:end * upsample]
            wav_list.append(sub_wav)

        # generate new mel chunk
        valid_start = valid_end
        valid_end = min(valid_start + block, mel_len)
        if valid_start - pad < 0:
            actual_start = 0
        else:
            actual_start = valid_start - pad
        actual_end = min(valid_end + pad, mel_len)
        mel_chunk = mel_streaming[actual_start:actual_end, :]

    wav = paddle.concat(wav_list, axis=0)
    wav_streaming = wav


@paddle.no_grad()
# 非流式AM / 流式AM + 非流式Voc
def am_nonstreaming_voc(args, am_infer_info, voc_infer_info, part_phone_ids,
                        part_tone_ids):
    mel = gen_mel(args, am_infer_info, part_phone_ids, part_tone_ids)
    am_infer_time = time.time()
    voc_inference, voc_config = voc_infer_info
    wav = voc_inference(mel)
    first_response_time = time.time()
    final_response_time = first_response_time
    voc_infer_time = first_response_time

    return am_infer_time, voc_infer_time, first_response_time, final_response_time, wav


@paddle.no_grad()
# 非流式AM + 流式Voc
def nonstreaming_am_streaming_voc(args, am_infer_info, voc_infer_info,
                                  part_phone_ids, part_tone_ids):
    global mel_streaming
    global streaming_first_time
    global wav_streaming

    mel = gen_mel(args, am_infer_info, part_phone_ids, part_tone_ids)
    am_infer_time = time.time()

    # voc streaming
    mel_streaming = mel
    mel_len = mel.shape[0]
    streaming_voc_infer(args, voc_infer_info, mel_len)
    first_response_time = streaming_first_time
    wav = wav_streaming
    final_response_time = time.time()
    voc_infer_time = final_response_time

    return am_infer_time, voc_infer_time, first_response_time, final_response_time, wav


@paddle.no_grad()
# 流式AM + 流式 Voc
def streaming_am_streaming_voc(args, am_infer_info, voc_infer_info,
                               part_phone_ids, part_tone_ids):
    global mel_streaming
    global streaming_first_time
    global wav_streaming
    global streaming_voc_st
    mel_streaming = None
    #用来表示开启流式voc的线程
    flag = 1

    am, am_mu, am_std, am_config = am_infer_info
    orig_hs, h_masks = am.encoder_infer(part_phone_ids)
    mel_len = orig_hs.shape[1]
    am_block = args.am_block
    am_pad = args.am_pad
    hss = get_chunks(orig_hs, am_block, am_pad, "am")
    chunk_num = len(hss)

    for i, hs in enumerate(hss):
        before_outs, _ = am.decoder(hs)
        after_outs = before_outs + am.postnet(
            before_outs.transpose((0, 2, 1))).transpose((0, 2, 1))
        normalized_mel = after_outs[0]
        sub_mel = denorm(normalized_mel, am_mu, am_std)
        # clip output part of pad
        if i == 0:
            sub_mel = sub_mel[:-am_pad]
            mel_streaming = sub_mel
        elif i == chunk_num - 1:
            # 最后一块的右侧一定没有 pad 够
            sub_mel = sub_mel[am_pad:]
            mel_streaming = paddle.concat([mel_streaming, sub_mel])
            am_infer_time = time.time()
        else:
            # 倒数几块的右侧也可能没有 pad 够
            sub_mel = sub_mel[am_pad:(am_block + am_pad) - sub_mel.shape[0]]
            mel_streaming = paddle.concat([mel_streaming, sub_mel])

        if flag and mel_streaming.shape[0] > args.voc_block + args.voc_pad:
            t = threading.Thread(
                target=streaming_voc_infer,
                args=(args, voc_infer_info, mel_len, ))
            t.start()
            streaming_voc_st = time.time()
            flag = 0

    t.join()
    final_response_time = time.time()
    voc_infer_time = final_response_time
    first_response_time = streaming_first_time
    wav = wav_streaming

    return am_infer_time, voc_infer_time, first_response_time, final_response_time, wav


def warm_up(args, logger, frontend, am_infer_info, voc_infer_info):
    global sample_rate
    logger.info(
        "Before the formal test, we test a few texts to make the inference speed more stable."
    )
    if args.lang == 'zh':
        sentence = "您好，欢迎使用语音合成服务。"
    if args.lang == 'en':
        sentence = "Hello and welcome to the speech synthesis service."

    if args.voc_streaming:
        if args.am_streaming:
            infer_func = streaming_am_streaming_voc
        else:
            infer_func = nonstreaming_am_streaming_voc
    else:
        infer_func = am_nonstreaming_voc

    merge_sentences = True
    get_tone_ids = False
    for i in range(5):  # 推理5次
        st = time.time()
        phone_ids, tone_ids = get_phone(args, frontend, sentence,
                                        merge_sentences, get_tone_ids)
        part_phone_ids = phone_ids[0]
        if tone_ids:
            part_tone_ids = tone_ids[0]
        else:
            part_tone_ids = None

        am_infer_time, voc_infer_time, first_response_time, final_response_time, wav = infer_func(
            args, am_infer_info, voc_infer_info, part_phone_ids, part_tone_ids)
        wav = wav.numpy()
        duration = wav.size / sample_rate
        logger.info(
            f"sentence: {sentence}; duration: {duration} s; first response time: {first_response_time - st} s; final response time: {final_response_time - st} s"
        )


def evaluate(args, logger, frontend, am_infer_info, voc_infer_info):
    global sample_rate
    sentences = get_sentences(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    get_tone_ids = False
    merge_sentences = True

    # choose infer function
    if args.voc_streaming:
        if args.am_streaming:
            infer_func = streaming_am_streaming_voc
        else:
            infer_func = nonstreaming_am_streaming_voc
    else:
        infer_func = am_nonstreaming_voc

    final_up_duration = 0.0
    sentence_count = 0
    front_time_list = []
    am_time_list = []
    voc_time_list = []
    first_response_list = []
    final_response_list = []
    sentence_length_list = []
    duration_list = []

    for utt_id, sentence in sentences:
        # front
        front_st = time.time()
        phone_ids, tone_ids = get_phone(args, frontend, sentence,
                                        merge_sentences, get_tone_ids)
        part_phone_ids = phone_ids[0]
        if tone_ids:
            part_tone_ids = tone_ids[0]
        else:
            part_tone_ids = None
        front_et = time.time()
        front_time = front_et - front_st

        am_st = time.time()
        am_infer_time, voc_infer_time, first_response_time, final_response_time, wav = infer_func(
            args, am_infer_info, voc_infer_info, part_phone_ids, part_tone_ids)
        am_time = am_infer_time - am_st
        if args.voc_streaming and args.am_streaming:
            voc_time = voc_infer_time - streaming_voc_st
        else:
            voc_time = voc_infer_time - am_infer_time

        first_response = first_response_time - front_st
        final_response = final_response_time - front_st

        wav = wav.numpy()
        duration = wav.size / sample_rate
        sf.write(
            str(output_dir / (utt_id + ".wav")), wav, samplerate=sample_rate)
        print(f"{utt_id} done!")

        sentence_count += 1
        front_time_list.append(front_time)
        am_time_list.append(am_time)
        voc_time_list.append(voc_time)
        first_response_list.append(first_response)
        final_response_list.append(final_response)
        sentence_length_list.append(len(sentence))
        duration_list.append(duration)

        logger.info(
            f"uttid: {utt_id}; sentence: '{sentence}'; front time: {front_time} s; am time: {am_time} s; voc time: {voc_time} s; \
                        first response time: {first_response} s; final response time: {final_response} s; audio duration: {duration} s;"
        )

        if final_response > duration:
            final_up_duration += 1

    all_time_sum = sum(final_response_list)
    front_rate = sum(front_time_list) / all_time_sum
    am_rate = sum(am_time_list) / all_time_sum
    voc_rate = sum(voc_time_list) / all_time_sum
    rtf = all_time_sum / sum(duration_list)

    logger.info(
        f"The length of test text information, test num: {sentence_count}; text num: {sum(sentence_length_list)}; min: {min(sentence_length_list)}; max: {max(sentence_length_list)}; avg: {sum(sentence_length_list)/len(sentence_length_list)}"
    )
    logger.info(
        f"duration information, min: {min(duration_list)}; max: {max(duration_list)}; avg: {sum(duration_list) / len(duration_list)}; sum: {sum(duration_list)}"
    )
    logger.info(
        f"Front time information: min: {min(front_time_list)} s; max: {max(front_time_list)} s; avg: {sum(front_time_list)/len(front_time_list)} s; ratio: {front_rate * 100}%"
    )
    logger.info(
        f"AM time information: min: {min(am_time_list)} s; max: {max(am_time_list)} s; avg: {sum(am_time_list)/len(am_time_list)} s; ratio: {am_rate * 100}%"
    )
    logger.info(
        f"Vocoder time information: min: {min(voc_time_list)} s, max: {max(voc_time_list)} s; avg: {sum(voc_time_list)/len(voc_time_list)} s; ratio: {voc_rate * 100}%"
    )
    logger.info(
        f"first response time information: min: {min(first_response_list)} s; max: {max(first_response_list)} s; avg: {sum(first_response_list)/len(first_response_list)} s"
    )
    logger.info(
        f"final response time information: min: {min(final_response_list)} s; max: {max(final_response_list)} s; avg: {sum(final_response_list)/len(final_response_list)} s"
    )
    logger.info(f"RTF is: {rtf}")
    logger.info(
        f"The number of final_response is greater than duration is {final_up_duration}, ratio: {final_up_duration / sentence_count}%"
    )


def parse_args():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(
        description="Synthesize with acoustic model & vocoder")
    # acoustic model
    parser.add_argument(
        '--am',
        type=str,
        default='fastspeech2_csmsc',
        choices=['fastspeech2_csmsc', 'fastspeech2_cnndecoder_csmsc'],
        help='Choose acoustic model type of tts task. where fastspeech2_cnndecoder_csmsc supports streaming inference'
    )

    parser.add_argument(
        '--am_config',
        type=str,
        default=None,
        help='Config of acoustic model. Use deault config when it is None.')
    parser.add_argument(
        '--am_ckpt',
        type=str,
        default=None,
        help='Checkpoint file of acoustic model.')
    parser.add_argument(
        "--am_stat",
        type=str,
        default=None,
        help="mean and standard deviation used to normalize spectrogram when training acoustic model."
    )
    parser.add_argument(
        "--phones_dict", type=str, default=None, help="phone vocabulary file.")
    parser.add_argument(
        "--tones_dict", type=str, default=None, help="tone vocabulary file.")
    # vocoder
    parser.add_argument(
        '--voc',
        type=str,
        default='mb_melgan_csmsc',
        choices=['mb_melgan_csmsc', 'hifigan_csmsc'],
        help='Choose vocoder type of tts task.')
    parser.add_argument(
        '--voc_config',
        type=str,
        default=None,
        help='Config of voc. Use deault config when it is None.')
    parser.add_argument(
        '--voc_ckpt', type=str, default=None, help='Checkpoint file of voc.')
    parser.add_argument(
        "--voc_stat",
        type=str,
        default=None,
        help="mean and standard deviation used to normalize spectrogram when training voc."
    )
    # other
    parser.add_argument(
        '--lang',
        type=str,
        default='zh',
        choices=['zh', 'en'],
        help='Choose model language. zh or en')

    parser.add_argument(
        "--device", type=str, default='cpu', help="set cpu or gpu:id")

    parser.add_argument(
        "--text",
        type=str,
        default="./csmsc_test.txt",
        help="text to synthesize, a 'utt_id sentence' pair per line.")
    parser.add_argument("--output_dir", type=str, help="output dir.")
    parser.add_argument(
        "--log_file", type=str, default="result.log", help="log file.")

    parser.add_argument(
        "--am_streaming",
        type=str2bool,
        default=False,
        help="whether use streaming acoustic model")

    parser.add_argument("--am_pad", type=int, default=12, help="am pad size.")

    parser.add_argument(
        "--am_block", type=int, default=42, help="am block size.")

    parser.add_argument(
        "--voc_streaming",
        type=str2bool,
        default=False,
        help="whether use streaming vocoder model")

    parser.add_argument("--voc_pad", type=int, default=14, help="voc pad size.")

    parser.add_argument(
        "--voc_block", type=int, default=14, help="voc block size.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    paddle.set_device(args.device)
    if args.am_streaming:
        assert (args.am == 'fastspeech2_cnndecoder_csmsc')

    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=args.log_file, mode='w')
    formatter = logging.Formatter(
        '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    )
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.DEBUG)

    # set basic information
    logger.info(
        f"AM: {args.am}; Vocoder: {args.voc}; device: {args.device}; am streaming: {args.am_streaming}; voc streaming: {args.voc_streaming}"
    )
    logger.info(
        f"am pad size: {args.am_pad}; am block size: {args.am_block}; voc pad size: {args.voc_pad}; voc block size: {args.voc_block};"
    )

    # get information about model
    frontend, am_infer_info, voc_infer_info = init(args)
    logger.info(
        "************************ warm up *********************************")
    warm_up(args, logger, frontend, am_infer_info, voc_infer_info)
    logger.info(
        "************************ normal test *******************************")
    evaluate(args, logger, frontend, am_infer_info, voc_infer_info)


if __name__ == "__main__":
    main()
