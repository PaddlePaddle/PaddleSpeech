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
import os
from pathlib import Path

import soundfile as sf
from paddle import inference

from paddlespeech.t2s.frontend.zh_frontend import Frontend


def main():
    parser = argparse.ArgumentParser(
        description="Paddle Infernce with speedyspeech & parallel wavegan.")
    parser.add_argument(
        "--inference-dir", type=str, help="dir to save inference models")
    parser.add_argument(
        "--text",
        type=str,
        help="text to synthesize, a 'utt_id sentence' pair per line")
    parser.add_argument("--output-dir", type=str, help="output dir")
    parser.add_argument(
        "--enable-auto-log", action="store_true", help="use auto log")
    parser.add_argument(
        "--phones-dict",
        type=str,
        default="phones.txt",
        help="phone vocabulary file.")

    args, _ = parser.parse_known_args()

    frontend = Frontend(phone_vocab_path=args.phones_dict)
    print("frontend done!")

    fastspeech2_config = inference.Config(
        str(Path(args.inference_dir) / "fastspeech2.pdmodel"),
        str(Path(args.inference_dir) / "fastspeech2.pdiparams"))
    fastspeech2_config.enable_use_gpu(50, 0)
    # This line must be commented, if not, it will OOM
    # fastspeech2_config.enable_memory_optim()
    fastspeech2_predictor = inference.create_predictor(fastspeech2_config)

    pwg_config = inference.Config(
        str(Path(args.inference_dir) / "pwg.pdmodel"),
        str(Path(args.inference_dir) / "pwg.pdiparams"))
    pwg_config.enable_use_gpu(100, 0)
    pwg_config.enable_memory_optim()
    pwg_predictor = inference.create_predictor(pwg_config)

    if args.enable_auto_log:
        import auto_log
        os.makedirs("output", exist_ok=True)
        pid = os.getpid()
        logger = auto_log.AutoLogger(
            model_name="fastspeech2",
            model_precision='float32',
            batch_size=1,
            data_shape="dynamic",
            save_path="./output/auto_log.log",
            inference_config=fastspeech2_config,
            pids=pid,
            process_name=None,
            gpu_ids=0,
            time_keys=['preprocess_time', 'inference_time', 'postprocess_time'],
            warmup=0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sentences = []

    with open(args.text, 'rt') as f:
        for line in f:
            items = line.strip().split()
            utt_id = items[0]
            sentence = "".join(items[1:])
            sentences.append((utt_id, sentence))

    for utt_id, sentence in sentences:
        if args.enable_auto_log:
            logger.times.start()
        input_ids = frontend.get_input_ids(sentence, merge_sentences=True)
        phone_ids = input_ids["phone_ids"]
        phones = phone_ids[0].numpy()

        if args.enable_auto_log:
            logger.times.stamp()

        input_names = fastspeech2_predictor.get_input_names()
        phones_handle = fastspeech2_predictor.get_input_handle(input_names[0])

        phones_handle.reshape(phones.shape)
        phones_handle.copy_from_cpu(phones)

        fastspeech2_predictor.run()
        output_names = fastspeech2_predictor.get_output_names()
        output_handle = fastspeech2_predictor.get_output_handle(output_names[0])
        output_data = output_handle.copy_to_cpu()

        input_names = pwg_predictor.get_input_names()
        mel_handle = pwg_predictor.get_input_handle(input_names[0])
        mel_handle.reshape(output_data.shape)
        mel_handle.copy_from_cpu(output_data)

        pwg_predictor.run()
        output_names = pwg_predictor.get_output_names()
        output_handle = pwg_predictor.get_output_handle(output_names[0])
        wav = output_data = output_handle.copy_to_cpu()

        if args.enable_auto_log:
            logger.times.stamp()

        sf.write(output_dir / (utt_id + ".wav"), wav, samplerate=24000)

        if args.enable_auto_log:
            logger.times.end(stamp=True)
        print(f"{utt_id} done!")

    if args.enable_auto_log:
        logger.report()


if __name__ == "__main__":
    main()
