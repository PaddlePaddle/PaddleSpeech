# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from concurrent.futures import ThreadPoolExecutor
from operator import itemgetter
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

import jsonlines
import librosa
import numpy as np
import tqdm
import yaml
from yacs.config import CfgNode

from paddlespeech.t2s.datasets.get_feats import LogMelFBank
from paddlespeech.t2s.datasets.preprocess_utils import get_spk_id_map

speaker_set = set()


def process_sentence(config: Dict[str, Any],
                     fp: Path,
                     output_dir: Path,
                     mel_extractor=None):
    utt_id = fp.stem
    # for vctk
    if utt_id.endswith("_mic2"):
        utt_id = utt_id[:-5]
        speaker = utt_id.split('_')[0]
        speaker_set.add(speaker)
    # 需要额外获取 speaker
    record = None
    # reading, resampling may occur
    # 源码的 bug, 读取的时候按照 24000 读取，但是提取 mel 的时候按照 16000 提取
    # 具体参考 https://github.com/PaddlePaddle/PaddleSpeech/blob/c7d24ba42c377fe4c0765c6b1faa202a9aeb136f/paddlespeech/t2s/exps/starganv2_vc/vc.py#L165
    # 之后需要换成按照 24000 读取和按照 24000 提取 mel
    wav, _ = librosa.load(str(fp), sr=24000)
    max_value = np.abs(wav).max()
    if max_value > 1.0:
        wav = wav / max_value
    assert len(wav.shape) == 1, f"{utt_id} is not a mono-channel audio."
    assert np.abs(
        wav).max() <= 1.0, f"{utt_id} is seems to be different that 16 bit PCM."
    # extract mel feats
    # 注意这里 base = 'e', 后续需要换成 base='10', 我们其他 TTS 模型都是 base='10'
    logmel = mel_extractor.get_log_mel_fbank(wav, base='e')
    mel_path = output_dir / (utt_id + "_speech.npy")
    np.save(mel_path, logmel)
    record = {"utt_id": utt_id, "speech": str(mel_path), "speaker": speaker}
    return record


def process_sentences(
        config,
        fps: List[Path],
        output_dir: Path,
        mel_extractor=None,
        nprocs: int=1, ):
    if nprocs == 1:
        results = []
        for fp in tqdm.tqdm(fps, total=len(fps)):
            record = process_sentence(
                config=config,
                fp=fp,
                output_dir=output_dir,
                mel_extractor=mel_extractor)
            if record:
                results.append(record)
    else:
        with ThreadPoolExecutor(nprocs) as pool:
            futures = []
            with tqdm.tqdm(total=len(fps)) as progress:
                for fp in fps:
                    future = pool.submit(process_sentence, config, fp,
                                         output_dir, mel_extractor)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)

                results = []
                for ft in futures:
                    record = ft.result()
                    if record:
                        results.append(record)

    results.sort(key=itemgetter("utt_id"))
    with jsonlines.open(output_dir / "metadata.jsonl", 'w') as writer:
        for item in results:
            writer.write(item)
    print("Done")


def main():
    # parse config and args
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features.")

    parser.add_argument(
        "--dataset",
        default="vctk",
        type=str,
        help="name of dataset, should in {vctk} now")

    parser.add_argument(
        "--rootdir", default=None, type=str, help="directory to dataset.")

    parser.add_argument(
        "--dumpdir",
        type=str,
        required=True,
        help="directory to dump feature files.")

    parser.add_argument("--config", type=str, help="StarGANv2VC config file.")

    parser.add_argument(
        "--num-cpu", type=int, default=1, help="number of process.")

    args = parser.parse_args()

    rootdir = Path(args.rootdir).expanduser()
    dumpdir = Path(args.dumpdir).expanduser()
    # use absolute path
    dumpdir = dumpdir.resolve()
    dumpdir.mkdir(parents=True, exist_ok=True)

    assert rootdir.is_dir()

    with open(args.config, 'rt') as f:
        config = CfgNode(yaml.safe_load(f))

    if args.dataset == "vctk":
        sub_num_dev = 5
        wav_dir = rootdir / "wav48_silence_trimmed"
        train_wav_files = []
        dev_wav_files = []
        test_wav_files = []
        # only for test
        for speaker in os.listdir(wav_dir):
            wav_files = sorted(list((wav_dir / speaker).rglob("*_mic2.flac")))
            if len(wav_files) > 100:
                train_wav_files += wav_files[:-sub_num_dev * 2]
                dev_wav_files += wav_files[-sub_num_dev * 2:-sub_num_dev]
                test_wav_files += wav_files[-sub_num_dev:]
            else:
                train_wav_files += wav_files

    else:
        print("dataset should in {vctk} now!")

    train_dump_dir = dumpdir / "train" / "raw"
    train_dump_dir.mkdir(parents=True, exist_ok=True)
    dev_dump_dir = dumpdir / "dev" / "raw"
    dev_dump_dir.mkdir(parents=True, exist_ok=True)
    test_dump_dir = dumpdir / "test" / "raw"
    test_dump_dir.mkdir(parents=True, exist_ok=True)

    # Extractor
    mel_extractor = LogMelFBank(
        sr=config.fs,
        n_fft=config.n_fft,
        hop_length=config.n_shift,
        win_length=config.win_length,
        window=config.window,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
        # None here
        norm=config.norm,
        htk=config.htk,
        power=config.power)

    # process for the 3 sections
    if train_wav_files:
        process_sentences(
            config=config,
            fps=train_wav_files,
            output_dir=train_dump_dir,
            mel_extractor=mel_extractor,
            nprocs=args.num_cpu)
    if dev_wav_files:
        process_sentences(
            config=config,
            fps=dev_wav_files,
            output_dir=dev_dump_dir,
            mel_extractor=mel_extractor,
            nprocs=args.num_cpu)
    if test_wav_files:
        process_sentences(
            config=config,
            fps=test_wav_files,
            output_dir=test_dump_dir,
            mel_extractor=mel_extractor,
            nprocs=args.num_cpu)

    speaker_id_map_path = dumpdir / "speaker_id_map.txt"
    get_spk_id_map(speaker_set, speaker_id_map_path)


if __name__ == "__main__":
    main()
