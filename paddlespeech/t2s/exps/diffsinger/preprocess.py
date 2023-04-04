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

from paddlespeech.t2s.datasets.get_feats import Energy
from paddlespeech.t2s.datasets.get_feats import LogMelFBank
from paddlespeech.t2s.datasets.get_feats import Pitch
from paddlespeech.t2s.datasets.preprocess_utils import compare_duration_and_mel_length
from paddlespeech.t2s.datasets.preprocess_utils import get_input_token
from paddlespeech.t2s.datasets.preprocess_utils import get_sentences_svs
from paddlespeech.t2s.datasets.preprocess_utils import get_spk_id_map
from paddlespeech.t2s.utils import str2bool

ALL_INITIALS = [
    'zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h',
    'j', 'q', 'x', 'r', 'z', 'c', 's', 'y', 'w'
]
ALL_FINALS = [
    'a', 'ai', 'an', 'ang', 'ao', 'e', 'ei', 'en', 'eng', 'er', 'i', 'ia',
    'ian', 'iang', 'iao', 'ie', 'in', 'ing', 'iong', 'iu', 'ng', 'o', 'ong',
    'ou', 'u', 'ua', 'uai', 'uan', 'uang', 'ui', 'un', 'uo', 'v', 'van', 've',
    'vn'
]


def process_sentence(
    config: Dict[str, Any],
    fp: Path,
    sentences: Dict,
    output_dir: Path,
    mel_extractor=None,
    pitch_extractor=None,
    energy_extractor=None,
    cut_sil: bool = True,
    spk_emb_dir: Path = None,
):
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
        note = sentences[utt_id][2]
        note_dur = sentences[utt_id][3]
        is_slur = sentences[utt_id][4]
        speaker = sentences[utt_id][-1]

        # extract mel feats
        logmel = mel_extractor.get_log_mel_fbank(wav)
        # change duration according to mel_length
        compare_duration_and_mel_length(sentences, utt_id, logmel)
        # utt_id may be popped in compare_duration_and_mel_length
        if utt_id not in sentences:
            return None
        phones = sentences[utt_id][0]
        durations = sentences[utt_id][1]
        num_frames = logmel.shape[0]

        assert sum(
            durations
        ) == num_frames, "the sum of durations doesn't equal to the num of mel frames. "
        speech_dir = output_dir / "data_speech"
        speech_dir.mkdir(parents=True, exist_ok=True)
        speech_path = speech_dir / (utt_id + "_speech.npy")
        np.save(speech_path, logmel)
        # extract pitch and energy
        pitch = pitch_extractor.get_pitch(wav)
        assert pitch.shape[0] == num_frames
        pitch_dir = output_dir / "data_pitch"
        pitch_dir.mkdir(parents=True, exist_ok=True)
        pitch_path = pitch_dir / (utt_id + "_pitch.npy")
        np.save(pitch_path, pitch)
        energy = energy_extractor.get_energy(wav)
        assert energy.shape[0] == num_frames
        energy_dir = output_dir / "data_energy"
        energy_dir.mkdir(parents=True, exist_ok=True)
        energy_path = energy_dir / (utt_id + "_energy.npy")
        np.save(energy_path, energy)

        record = {
            "utt_id": utt_id,
            "phones": phones,
            "text_lengths": len(phones),
            "speech_lengths": num_frames,
            "durations": durations,
            "speech": str(speech_path),
            "pitch": str(pitch_path),
            "energy": str(energy_path),
            "speaker": speaker,
            "note": note,
            "note_dur": note_dur,
            "is_slur": is_slur,
        }
        if spk_emb_dir:
            if speaker in os.listdir(spk_emb_dir):
                embed_name = utt_id + ".npy"
                embed_path = spk_emb_dir / speaker / embed_name
                if embed_path.is_file():
                    record["spk_emb"] = str(embed_path)
                else:
                    return None
    return record


def process_sentences(
    config,
    fps: List[Path],
    sentences: Dict,
    output_dir: Path,
    mel_extractor=None,
    pitch_extractor=None,
    energy_extractor=None,
    nprocs: int = 1,
    cut_sil: bool = True,
    spk_emb_dir: Path = None,
    write_metadata_method: str = 'w',
):
    if nprocs == 1:
        results = []
        for fp in tqdm.tqdm(fps, total=len(fps)):
            record = process_sentence(
                config=config,
                fp=fp,
                sentences=sentences,
                output_dir=output_dir,
                mel_extractor=mel_extractor,
                pitch_extractor=pitch_extractor,
                energy_extractor=energy_extractor,
                cut_sil=cut_sil,
                spk_emb_dir=spk_emb_dir,
            )
            if record:
                results.append(record)
    else:
        with ThreadPoolExecutor(nprocs) as pool:
            futures = []
            with tqdm.tqdm(total=len(fps)) as progress:
                for fp in fps:
                    future = pool.submit(
                        process_sentence,
                        config,
                        fp,
                        sentences,
                        output_dir,
                        mel_extractor,
                        pitch_extractor,
                        energy_extractor,
                        cut_sil,
                        spk_emb_dir,
                    )
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)

                results = []
                for ft in futures:
                    record = ft.result()
                    if record:
                        results.append(record)

    results.sort(key=itemgetter("utt_id"))
    with jsonlines.open(output_dir / "metadata.jsonl",
                        write_metadata_method) as writer:
        for item in results:
            writer.write(item)
    print("Done")


def main():
    # parse config and args
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features.")

    parser.add_argument("--dataset",
                        default="opencpop",
                        type=str,
                        help="name of dataset, should in {opencpop} now")

    parser.add_argument("--rootdir",
                        default=None,
                        type=str,
                        help="directory to dataset.")

    parser.add_argument("--dumpdir",
                        type=str,
                        required=True,
                        help="directory to dump feature files.")

    parser.add_argument("--label-file",
                        default=None,
                        type=str,
                        help="path to label file.")

    parser.add_argument("--config", type=str, help="diffsinger config file.")

    parser.add_argument("--num-cpu",
                        type=int,
                        default=1,
                        help="number of process.")

    parser.add_argument("--cut-sil",
                        type=str2bool,
                        default=True,
                        help="whether cut sil in the edge of audio")

    parser.add_argument("--spk_emb_dir",
                        default=None,
                        type=str,
                        help="directory to speaker embedding files.")

    parser.add_argument("--write_metadata_method",
                        default="w",
                        type=str,
                        choices=["w", "a"],
                        help="How the metadata.jsonl file is written.")
    args = parser.parse_args()

    rootdir = Path(args.rootdir).expanduser()
    dumpdir = Path(args.dumpdir).expanduser()
    # use absolute path
    dumpdir = dumpdir.resolve()
    dumpdir.mkdir(parents=True, exist_ok=True)
    label_file = Path(args.label_file).expanduser()

    if args.spk_emb_dir:
        spk_emb_dir = Path(args.spk_emb_dir).expanduser().resolve()
    else:
        spk_emb_dir = None

    assert rootdir.is_dir()
    assert label_file.is_file()

    with open(args.config, 'rt') as f:
        config = CfgNode(yaml.safe_load(f))

    sentences, speaker_set = get_sentences_svs(
        label_file,
        dataset=args.dataset,
        sample_rate=config.fs,
        n_shift=config.n_shift,
    )

    phone_id_map_path = dumpdir / "phone_id_map.txt"
    speaker_id_map_path = dumpdir / "speaker_id_map.txt"
    get_input_token(sentences, phone_id_map_path, args.dataset)
    get_spk_id_map(speaker_set, speaker_id_map_path)

    if args.dataset == "opencpop":
        wavdir = rootdir / "wavs"
        # split data into 3 sections
        train_file = rootdir / "train.txt"
        train_wav_files = []
        with open(train_file, "r") as f_train:
            for line in f_train.readlines():
                utt = line.split("|")[0]
                wav_name = utt + ".wav"
                wav_path = wavdir / wav_name
                train_wav_files.append(wav_path)

        test_file = rootdir / "test.txt"
        dev_wav_files = []
        test_wav_files = []
        num_dev = 106
        count = 0
        with open(test_file, "r") as f_test:
            for line in f_test.readlines():
                count += 1
                utt = line.split("|")[0]
                wav_name = utt + ".wav"
                wav_path = wavdir / wav_name
                if count > num_dev:
                    test_wav_files.append(wav_path)
                else:
                    dev_wav_files.append(wav_path)

    else:
        print("dataset should in {opencpop} now!")

    train_dump_dir = dumpdir / "train" / "raw"
    train_dump_dir.mkdir(parents=True, exist_ok=True)
    dev_dump_dir = dumpdir / "dev" / "raw"
    dev_dump_dir.mkdir(parents=True, exist_ok=True)
    test_dump_dir = dumpdir / "test" / "raw"
    test_dump_dir.mkdir(parents=True, exist_ok=True)

    # Extractor
    mel_extractor = LogMelFBank(sr=config.fs,
                                n_fft=config.n_fft,
                                hop_length=config.n_shift,
                                win_length=config.win_length,
                                window=config.window,
                                n_mels=config.n_mels,
                                fmin=config.fmin,
                                fmax=config.fmax)
    pitch_extractor = Pitch(sr=config.fs,
                            hop_length=config.n_shift,
                            f0min=config.f0min,
                            f0max=config.f0max)
    energy_extractor = Energy(n_fft=config.n_fft,
                              hop_length=config.n_shift,
                              win_length=config.win_length,
                              window=config.window)

    # process for the 3 sections
    if train_wav_files:
        process_sentences(config=config,
                          fps=train_wav_files,
                          sentences=sentences,
                          output_dir=train_dump_dir,
                          mel_extractor=mel_extractor,
                          pitch_extractor=pitch_extractor,
                          energy_extractor=energy_extractor,
                          nprocs=args.num_cpu,
                          cut_sil=args.cut_sil,
                          spk_emb_dir=spk_emb_dir,
                          write_metadata_method=args.write_metadata_method)
    if dev_wav_files:
        process_sentences(config=config,
                          fps=dev_wav_files,
                          sentences=sentences,
                          output_dir=dev_dump_dir,
                          mel_extractor=mel_extractor,
                          pitch_extractor=pitch_extractor,
                          energy_extractor=energy_extractor,
                          cut_sil=args.cut_sil,
                          spk_emb_dir=spk_emb_dir,
                          write_metadata_method=args.write_metadata_method)
    if test_wav_files:
        process_sentences(config=config,
                          fps=test_wav_files,
                          sentences=sentences,
                          output_dir=test_dump_dir,
                          mel_extractor=mel_extractor,
                          pitch_extractor=pitch_extractor,
                          energy_extractor=energy_extractor,
                          nprocs=args.num_cpu,
                          cut_sil=args.cut_sil,
                          spk_emb_dir=spk_emb_dir,
                          write_metadata_method=args.write_metadata_method)


if __name__ == "__main__":
    main()
