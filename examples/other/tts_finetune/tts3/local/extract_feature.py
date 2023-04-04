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
import logging
import os
from operator import itemgetter
from pathlib import Path
from typing import Dict
from typing import Union

import jsonlines
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from yacs.config import CfgNode

from paddlespeech.t2s.datasets.data_table import DataTable
from paddlespeech.t2s.datasets.get_feats import Energy
from paddlespeech.t2s.datasets.get_feats import LogMelFBank
from paddlespeech.t2s.datasets.get_feats import Pitch
from paddlespeech.t2s.datasets.preprocess_utils import get_phn_dur
from paddlespeech.t2s.datasets.preprocess_utils import merge_silence
from paddlespeech.t2s.exps.fastspeech2.preprocess import process_sentences


def read_stats(stats_file: Union[str, Path]):
    scaler = StandardScaler()
    scaler.mean_ = np.load(stats_file)[0]
    scaler.scale_ = np.load(stats_file)[1]
    scaler.n_features_in_ = scaler.mean_.shape[0]
    return scaler


def get_stats(pretrained_model_dir: Path):
    speech_stats_file = pretrained_model_dir / "speech_stats.npy"
    pitch_stats_file = pretrained_model_dir / "pitch_stats.npy"
    energy_stats_file = pretrained_model_dir / "energy_stats.npy"
    speech_scaler = read_stats(speech_stats_file)
    pitch_scaler = read_stats(pitch_stats_file)
    energy_scaler = read_stats(energy_stats_file)

    return speech_scaler, pitch_scaler, energy_scaler


def get_map(duration_file: Union[str, Path],
            dump_dir: Path,
            pretrained_model_dir: Path,
            replace_spkid: int = 0):
    """get phone map and speaker map, save on dump_dir

    Args:
        duration_file (str): durantions.txt
        dump_dir (Path): dump dir
        pretrained_model_dir (Path): pretrained model dir
        replace_spkid (int): replace spk id 
    """
    # copy phone map file from pretrained model path
    phones_dict = dump_dir / "phone_id_map.txt"
    os.system("cp %s %s" %
              (pretrained_model_dir / "phone_id_map.txt", phones_dict))

    # create a new speaker map file, replace the previous speakers.
    sentences, speaker_set = get_phn_dur(duration_file)
    merge_silence(sentences)
    speakers = sorted(list(speaker_set))
    num = len(speakers)
    speaker_dict = dump_dir / "speaker_id_map.txt"
    spk_dict = {}
    # get raw spkid-spk dict
    with open(pretrained_model_dir / "speaker_id_map.txt", 'r') as fr:
        for line in fr.readlines():
            spk = line.strip().split(" ")[0]
            spk_id = line.strip().split(" ")[1]
            spk_dict[spk_id] = spk

    # replace spk on spkid-spk dict
    assert replace_spkid + num - 1 < len(
        spk_dict), "Please set correct replace spk id."
    for i, spk in enumerate(speakers):
        spk_dict[str(replace_spkid + i)] = spk

    # write a new spk map file
    with open(speaker_dict, 'w') as f:
        for spk_id in spk_dict.keys():
            f.write(spk_dict[spk_id] + ' ' + spk_id + '\n')

    vocab_phones = {}
    with open(phones_dict, 'rt') as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    for phn, id in phn_id:
        vocab_phones[phn] = int(id)

    vocab_speaker = {}
    with open(speaker_dict, 'rt') as f:
        spk_id = [line.strip().split() for line in f.readlines()]
    for spk, id in spk_id:
        vocab_speaker[spk] = int(id)

    return sentences, vocab_phones, vocab_speaker


def get_extractor(config):
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

    return mel_extractor, pitch_extractor, energy_extractor


def normalize(speech_scaler, pitch_scaler, energy_scaler, vocab_phones: Dict,
              vocab_speaker: Dict, raw_dump_dir: Path, type: str):

    dumpdir = raw_dump_dir / type / "norm"
    dumpdir = Path(dumpdir).expanduser()
    dumpdir.mkdir(parents=True, exist_ok=True)

    # get dataset
    metadata_file = raw_dump_dir / type / "raw" / "metadata.jsonl"
    with jsonlines.open(metadata_file, 'r') as reader:
        metadata = list(reader)
    dataset = DataTable(metadata,
                        converters={
                            "speech": np.load,
                            "pitch": np.load,
                            "energy": np.load,
                        })
    logging.info(f"The number of files = {len(dataset)}.")

    # process each file
    output_metadata = []

    for item in tqdm(dataset):
        utt_id = item['utt_id']
        speech = item['speech']
        pitch = item['pitch']
        energy = item['energy']
        # normalize
        speech = speech_scaler.transform(speech)
        speech_dir = dumpdir / "data_speech"
        speech_dir.mkdir(parents=True, exist_ok=True)
        speech_path = speech_dir / f"{utt_id}_speech.npy"
        np.save(speech_path, speech.astype(np.float32), allow_pickle=False)

        pitch = pitch_scaler.transform(pitch)
        pitch_dir = dumpdir / "data_pitch"
        pitch_dir.mkdir(parents=True, exist_ok=True)
        pitch_path = pitch_dir / f"{utt_id}_pitch.npy"
        np.save(pitch_path, pitch.astype(np.float32), allow_pickle=False)

        energy = energy_scaler.transform(energy)
        energy_dir = dumpdir / "data_energy"
        energy_dir.mkdir(parents=True, exist_ok=True)
        energy_path = energy_dir / f"{utt_id}_energy.npy"
        np.save(energy_path, energy.astype(np.float32), allow_pickle=False)

        phone_ids = [vocab_phones[p] for p in item['phones']]
        spk_id = vocab_speaker[item["speaker"]]
        record = {
            "utt_id": item['utt_id'],
            "spk_id": spk_id,
            "text": phone_ids,
            "text_lengths": item['text_lengths'],
            "speech_lengths": item['speech_lengths'],
            "durations": item['durations'],
            "speech": str(speech_path),
            "pitch": str(pitch_path),
            "energy": str(energy_path)
        }
        # add spk_emb for voice cloning
        if "spk_emb" in item:
            record["spk_emb"] = str(item["spk_emb"])

        output_metadata.append(record)
    output_metadata.sort(key=itemgetter('utt_id'))
    output_metadata_path = Path(dumpdir) / "metadata.jsonl"
    with jsonlines.open(output_metadata_path, 'w') as writer:
        for item in output_metadata:
            writer.write(item)
    logging.info(f"metadata dumped into {output_metadata_path}")


def extract_feature(duration_file: str,
                    config,
                    input_dir: Path,
                    dump_dir: Path,
                    pretrained_model_dir: Path,
                    replace_spkid: int = 0):

    sentences, vocab_phones, vocab_speaker = get_map(duration_file, dump_dir,
                                                     pretrained_model_dir,
                                                     replace_spkid)
    mel_extractor, pitch_extractor, energy_extractor = get_extractor(config)

    wav_files = sorted(list((input_dir).rglob("*.wav")))
    # split data into 3 sections, train: len(wav_files) - 2, dev: 1, test: 1
    num_train = len(wav_files) - 2
    num_dev = 1
    print(num_train, num_dev)

    train_wav_files = wav_files[:num_train]
    dev_wav_files = wav_files[num_train:num_train + num_dev]
    test_wav_files = wav_files[num_train + num_dev:]

    train_dump_dir = dump_dir / "train" / "raw"
    train_dump_dir.mkdir(parents=True, exist_ok=True)
    dev_dump_dir = dump_dir / "dev" / "raw"
    dev_dump_dir.mkdir(parents=True, exist_ok=True)
    test_dump_dir = dump_dir / "test" / "raw"
    test_dump_dir.mkdir(parents=True, exist_ok=True)

    # process for the 3 sections
    num_cpu = 4
    cut_sil = True
    spk_emb_dir = None
    write_metadata_method = "w"
    speech_scaler, pitch_scaler, energy_scaler = get_stats(pretrained_model_dir)

    if train_wav_files:
        process_sentences(config=config,
                          fps=train_wav_files,
                          sentences=sentences,
                          output_dir=train_dump_dir,
                          mel_extractor=mel_extractor,
                          pitch_extractor=pitch_extractor,
                          energy_extractor=energy_extractor,
                          nprocs=num_cpu,
                          cut_sil=cut_sil,
                          spk_emb_dir=spk_emb_dir,
                          write_metadata_method=write_metadata_method)
        # norm
        normalize(speech_scaler, pitch_scaler, energy_scaler, vocab_phones,
                  vocab_speaker, dump_dir, "train")

    if dev_wav_files:
        process_sentences(config=config,
                          fps=dev_wav_files,
                          sentences=sentences,
                          output_dir=dev_dump_dir,
                          mel_extractor=mel_extractor,
                          pitch_extractor=pitch_extractor,
                          energy_extractor=energy_extractor,
                          nprocs=num_cpu,
                          cut_sil=cut_sil,
                          spk_emb_dir=spk_emb_dir,
                          write_metadata_method=write_metadata_method)
        # norm
        normalize(speech_scaler, pitch_scaler, energy_scaler, vocab_phones,
                  vocab_speaker, dump_dir, "dev")

    if test_wav_files:
        process_sentences(config=config,
                          fps=test_wav_files,
                          sentences=sentences,
                          output_dir=test_dump_dir,
                          mel_extractor=mel_extractor,
                          pitch_extractor=pitch_extractor,
                          energy_extractor=energy_extractor,
                          nprocs=num_cpu,
                          cut_sil=cut_sil,
                          spk_emb_dir=spk_emb_dir,
                          write_metadata_method=write_metadata_method)

        # norm
        normalize(speech_scaler, pitch_scaler, energy_scaler, vocab_phones,
                  vocab_speaker, dump_dir, "test")


if __name__ == '__main__':
    # parse config and args
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features.")

    parser.add_argument("--duration_file",
                        type=str,
                        default="./durations.txt",
                        help="duration file")

    parser.add_argument("--input_dir",
                        type=str,
                        default="./input/baker_mini/newdir",
                        help="directory containing audio and label file")

    parser.add_argument("--dump_dir",
                        type=str,
                        default="./dump",
                        help="dump dir")

    parser.add_argument(
        "--pretrained_model_dir",
        type=str,
        default="./pretrained_models/fastspeech2_aishell3_ckpt_1.1.0",
        help="Path to pretrained model")

    parser.add_argument("--replace_spkid",
                        type=int,
                        default=0,
                        help="replace spk id")

    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    dump_dir = Path(args.dump_dir).expanduser()
    dump_dir.mkdir(parents=True, exist_ok=True)
    pretrained_model_dir = Path(args.pretrained_model_dir).expanduser()

    # read config
    config_file = pretrained_model_dir / "default.yaml"
    with open(config_file) as f:
        config = CfgNode(yaml.safe_load(f))

    extract_feature(duration_file=args.duration_file,
                    config=config,
                    input_dir=input_dir,
                    dump_dir=dump_dir,
                    pretrained_model_dir=pretrained_model_dir,
                    replace_spkid=args.replace_spkid)
