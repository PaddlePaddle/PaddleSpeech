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
import re
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
from paddlespeech.t2s.datasets.preprocess_utils import compare_duration_and_mel_length
from paddlespeech.t2s.datasets.preprocess_utils import get_phn_dur
from paddlespeech.t2s.datasets.preprocess_utils import get_phones_tones
from paddlespeech.t2s.datasets.preprocess_utils import get_spk_id_map
from paddlespeech.t2s.datasets.preprocess_utils import merge_silence
from paddlespeech.t2s.utils import str2bool


def process_sentence(config: Dict[str, Any],
                     fp: Path,
                     sentences: Dict,
                     output_dir: Path,
                     mel_extractor=None,
                     cut_sil: bool=True):
    utt_id = fp.stem
    record = None
    if utt_id in sentences:
        # reading, resampling may occur
        wav, _ = librosa.load(str(fp), sr=config.fs)
        if len(wav.shape) != 1:
            return record
        max_value = np.abs(wav).max()
        if max_value > 1.0:
            wav = wav / max_value
        assert len(wav.shape) == 1, f"{utt_id} is not a mono-channel audio."
        assert np.abs(wav).max(
        ) <= 1.0, f"{utt_id} is seems to be different that 16 bit PCM."
        phones = sentences[utt_id][0]
        durations = sentences[utt_id][1]
        speaker = sentences[utt_id][2]
        d_cumsum = np.pad(np.array(durations).cumsum(0), (1, 0), 'constant')
        # little imprecise than use *.TextGrid directly
        times = librosa.frames_to_time(
            d_cumsum, sr=config.fs, hop_length=config.n_shift)
        if cut_sil:
            start = 0
            end = d_cumsum[-1]
            if phones[0] == "sil" and len(durations) > 1:
                start = times[1]
                durations = durations[1:]
                phones = phones[1:]
            if phones[-1] == 'sil' and len(durations) > 1:
                end = times[-2]
                durations = durations[:-1]
                phones = phones[:-1]
            sentences[utt_id][0] = phones
            sentences[utt_id][1] = durations
            start, end = librosa.time_to_samples([start, end], sr=config.fs)
            wav = wav[start:end]

        # extract mel feats
        logmel = mel_extractor.get_log_mel_fbank(wav)
        # change duration according to mel_length
        compare_duration_and_mel_length(sentences, utt_id, logmel)
        # utt_id may be popped in compare_duration_and_mel_length
        if utt_id not in sentences:
            return None
        labels = sentences[utt_id][0]
        # extract phone and duration
        phones = []
        tones = []
        for label in labels:
            # split tone from finals
            match = re.match(r'^(\w+)([012345])$', label)
            if match:
                phones.append(match.group(1))
                tones.append(match.group(2))
            else:
                phones.append(label)
                tones.append('0')
        durations = sentences[utt_id][1]
        num_frames = logmel.shape[0]
        assert sum(durations) == num_frames
        assert len(phones) == len(tones) == len(durations)

        mel_path = output_dir / (utt_id + "_feats.npy")
        np.save(mel_path, logmel)  # (num_frames, n_mels)
        record = {
            "utt_id": utt_id,
            "phones": phones,
            "tones": tones,
            "speaker": speaker,
            "num_phones": len(phones),
            "num_frames": num_frames,
            "durations": durations,
            "feats": str(mel_path),  # Path object
        }
    return record


def process_sentences(config,
                      fps: List[Path],
                      sentences: Dict,
                      output_dir: Path,
                      mel_extractor=None,
                      nprocs: int=1,
                      cut_sil: bool=True,
                      use_relative_path: bool=False):

    if nprocs == 1:
        results = []
        for fp in tqdm.tqdm(fps, total=len(fps)):
            record = process_sentence(
                config=config,
                fp=fp,
                sentences=sentences,
                output_dir=output_dir,
                mel_extractor=mel_extractor,
                cut_sil=cut_sil)
            if record:
                results.append(record)
    else:
        with ThreadPoolExecutor(nprocs) as pool:
            futures = []
            with tqdm.tqdm(total=len(fps)) as progress:
                for fp in fps:
                    future = pool.submit(process_sentence, config, fp,
                                         sentences, output_dir, mel_extractor,
                                         cut_sil)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)

                results = []
                for ft in futures:
                    record = ft.result()
                    if record:
                        results.append(record)

    results.sort(key=itemgetter("utt_id"))
    output_dir = Path(output_dir)
    metadata_path = output_dir / "metadata.jsonl"
    # NOTE: use relative path to the meta jsonlines file for Full Chain Project
    with jsonlines.open(metadata_path, 'w') as writer:
        for item in results:
            if use_relative_path:
                item["feats"] = str(Path(item["feats"]).relative_to(output_dir))
            writer.write(item)
    print("Done")


def main():
    # parse config and args
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features.")

    parser.add_argument(
        "--dataset",
        default="baker",
        type=str,
        help="name of dataset, should in {baker} now")

    parser.add_argument(
        "--rootdir", default=None, type=str, help="directory to dataset.")
    parser.add_argument(
        "--dumpdir",
        type=str,
        required=True,
        help="directory to dump feature files.")

    parser.add_argument(
        "--dur-file",
        default=None,
        type=str,
        help="path to baker durations.txt.")

    parser.add_argument("--config", type=str, help="fastspeech2 config file.")

    parser.add_argument(
        "--num-cpu", type=int, default=1, help="number of process.")

    parser.add_argument(
        "--cut-sil",
        type=str2bool,
        default=True,
        help="whether cut sil in the edge of audio")

    parser.add_argument(
        "--use-relative-path",
        type=str2bool,
        default=False,
        help="whether use relative path in metadata")

    args = parser.parse_args()

    rootdir = Path(args.rootdir).expanduser()
    dumpdir = Path(args.dumpdir).expanduser()
    # use absolute path
    dumpdir = dumpdir.resolve()
    dumpdir.mkdir(parents=True, exist_ok=True)
    dur_file = Path(args.dur_file).expanduser()

    assert rootdir.is_dir()
    assert dur_file.is_file()

    with open(args.config, 'rt') as f:
        config = CfgNode(yaml.safe_load(f))

    sentences, speaker_set = get_phn_dur(dur_file)

    merge_silence(sentences)
    phone_id_map_path = dumpdir / "phone_id_map.txt"
    tone_id_map_path = dumpdir / "tone_id_map.txt"
    get_phones_tones(sentences, phone_id_map_path, tone_id_map_path,
                     args.dataset)
    speaker_id_map_path = dumpdir / "speaker_id_map.txt"
    get_spk_id_map(speaker_set, speaker_id_map_path)

    if args.dataset == "baker":
        wav_files = sorted(list((rootdir / "Wave").rglob("*.wav")))
        # split data into 3 sections
        num_train = 9800
        num_dev = 100
        train_wav_files = wav_files[:num_train]
        dev_wav_files = wav_files[num_train:num_train + num_dev]
        test_wav_files = wav_files[num_train + num_dev:]

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
        fmax=config.fmax)

    # process for the 3 sections
    if train_wav_files:
        process_sentences(
            config=config,
            fps=train_wav_files,
            sentences=sentences,
            output_dir=train_dump_dir,
            mel_extractor=mel_extractor,
            nprocs=args.num_cpu,
            cut_sil=args.cut_sil,
            use_relative_path=args.use_relative_path)
    if dev_wav_files:
        process_sentences(
            config=config,
            fps=dev_wav_files,
            sentences=sentences,
            output_dir=dev_dump_dir,
            mel_extractor=mel_extractor,
            nprocs=args.num_cpu,
            cut_sil=args.cut_sil,
            use_relative_path=args.use_relative_path)
    if test_wav_files:
        process_sentences(
            config=config,
            fps=test_wav_files,
            sentences=sentences,
            output_dir=test_dump_dir,
            mel_extractor=mel_extractor,
            nprocs=args.num_cpu,
            cut_sil=args.cut_sil,
            use_relative_path=args.use_relative_path)


if __name__ == "__main__":
    main()
