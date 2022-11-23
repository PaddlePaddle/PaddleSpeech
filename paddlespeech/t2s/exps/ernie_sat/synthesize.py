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
from pathlib import Path

import jsonlines
import numpy as np
import paddle
import soundfile as sf
import yaml
from yacs.config import CfgNode

from paddlespeech.t2s.datasets.am_batch_fn import build_erniesat_collate_fn
from paddlespeech.t2s.exps.syn_utils import denorm
from paddlespeech.t2s.exps.syn_utils import get_am_inference
from paddlespeech.t2s.exps.syn_utils import get_test_dataset
from paddlespeech.t2s.exps.syn_utils import get_voc_inference


def evaluate(args):
    # dataloader has been too verbose
    logging.getLogger("DataLoader").disabled = True

    # construct dataset for evaluation
    with jsonlines.open(args.test_metadata, 'r') as reader:
        test_metadata = list(reader)

    # Init body.
    with open(args.erniesat_config) as f:
        erniesat_config = CfgNode(yaml.safe_load(f))
    with open(args.voc_config) as f:
        voc_config = CfgNode(yaml.safe_load(f))

    print("========Args========")
    print(yaml.safe_dump(vars(args)))
    print("========Config========")
    print(erniesat_config)
    print(voc_config)

    # ernie sat model
    erniesat_inference = get_am_inference(
        am='erniesat_dataset',
        am_config=erniesat_config,
        am_ckpt=args.erniesat_ckpt,
        am_stat=args.erniesat_stat,
        phones_dict=args.phones_dict)

    test_dataset = get_test_dataset(
        test_metadata=test_metadata, am='erniesat_dataset')

    # vocoder
    voc_inference = get_voc_inference(
        voc=args.voc,
        voc_config=voc_config,
        voc_ckpt=args.voc_ckpt,
        voc_stat=args.voc_stat)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    collate_fn = build_erniesat_collate_fn(
        mlm_prob=erniesat_config.mlm_prob,
        mean_phn_span=erniesat_config.mean_phn_span,
        seg_emb=erniesat_config.model['enc_input_layer'] == 'sega_mlm',
        text_masking=False)

    gen_raw = True
    erniesat_mu, erniesat_std = np.load(args.erniesat_stat)

    for datum in test_dataset:
        # collate function and dataloader
        utt_id = datum["utt_id"]
        speech_len = datum["speech_lengths"]

        # mask the middle 1/3 speech
        left_bdy, right_bdy = speech_len // 3, 2 * speech_len // 3
        span_bdy = [left_bdy, right_bdy]
        datum.update({"span_bdy": span_bdy})

        batch = collate_fn([datum])
        with paddle.no_grad():
            out_mels = erniesat_inference(
                speech=batch["speech"],
                text=batch["text"],
                masked_pos=batch["masked_pos"],
                speech_mask=batch["speech_mask"],
                text_mask=batch["text_mask"],
                speech_seg_pos=batch["speech_seg_pos"],
                text_seg_pos=batch["text_seg_pos"],
                span_bdy=span_bdy)

            # vocoder
            wav_list = []
            for mel in out_mels:
                part_wav = voc_inference(mel)
                wav_list.append(part_wav)
            wav = paddle.concat(wav_list)
            wav = wav.numpy()
            if gen_raw:
                speech = datum['speech']
                denorm_mel = denorm(speech, erniesat_mu, erniesat_std)
                denorm_mel = paddle.to_tensor(denorm_mel)
                wav_raw = voc_inference(denorm_mel)
                wav_raw = wav_raw.numpy()

        sf.write(
            str(output_dir / (utt_id + ".wav")),
            wav,
            samplerate=erniesat_config.fs)
        if gen_raw:
            sf.write(
                str(output_dir / (utt_id + "_raw" + ".wav")),
                wav_raw,
                samplerate=erniesat_config.fs)

        print(f"{utt_id} done!")


def parse_args():
    # parse args and config
    parser = argparse.ArgumentParser(
        description="Synthesize with acoustic model & vocoder")
    # ernie sat

    parser.add_argument(
        '--erniesat_config',
        type=str,
        default=None,
        help='Config of acoustic model.')
    parser.add_argument(
        '--erniesat_ckpt',
        type=str,
        default=None,
        help='Checkpoint file of acoustic model.')
    parser.add_argument(
        "--erniesat_stat",
        type=str,
        default=None,
        help="mean and standard deviation used to normalize spectrogram when training acoustic model."
    )
    parser.add_argument(
        "--phones_dict", type=str, default=None, help="phone vocabulary file.")
    # vocoder
    parser.add_argument(
        '--voc',
        type=str,
        default='pwgan_csmsc',
        choices=[
            'pwgan_aishell3',
            'pwgan_vctk',
            'hifigan_aishell3',
            'hifigan_vctk',
        ],
        help='Choose vocoder type of tts task.')
    parser.add_argument(
        '--voc_config', type=str, default=None, help='Config of voc.')
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
        "--ngpu", type=int, default=1, help="if ngpu == 0, use cpu.")
    parser.add_argument("--test_metadata", type=str, help="test metadata.")
    parser.add_argument("--output_dir", type=str, help="output dir.")

    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    if args.ngpu == 0:
        paddle.set_device("cpu")
    elif args.ngpu > 0:
        paddle.set_device("gpu")
    else:
        print("ngpu should >= 0 !")

    evaluate(args)


if __name__ == "__main__":
    main()
