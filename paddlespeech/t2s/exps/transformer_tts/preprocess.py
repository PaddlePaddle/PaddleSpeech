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
from yacs.config import CfgNode as Configuration

from paddlespeech.t2s.datasets.get_feats import LogMelFBank
from paddlespeech.t2s.frontend import English


def get_lj_sentences(file_name, frontend):
    '''read MFA duration.txt

    Args:
        file_name (str or Path)
    Returns:
        Dict: sentence: {'utt': ([char], [int])}
    '''
    f = open(file_name, 'r')
    sentence = {}
    speaker_set = set()
    for line in f:
        line_list = line.strip().split('|')
        utt = line_list[0]
        speaker = utt.split("-")[0][:2]
        speaker_set.add(speaker)
        raw_text = line_list[-1]
        phonemes = frontend.phoneticize(raw_text)
        phonemes = phonemes[1:-1]
        phonemes = [phn for phn in phonemes if not phn.isspace()]
        sentence[utt] = (phonemes, speaker)
    f.close()
    return sentence, speaker_set


def get_input_token(sentence, output_path):
    '''get phone set from training data and save it
    
    Args:
        sentence (Dict): sentence: {'utt': ([char], str)}
        output_path (str or path): path to save phone_id_map
    '''
    phn_token = set()
    for utt in sentence:
        for phn in sentence[utt][0]:
            if phn != "<eos>":
                phn_token.add(phn)
    phn_token = list(phn_token)
    phn_token.sort()
    phn_token = ["<pad>", "<unk>"] + phn_token
    phn_token += ["<eos>"]

    with open(output_path, 'w') as f:
        for i, phn in enumerate(phn_token):
            f.write(phn + ' ' + str(i) + '\n')


def get_spk_id_map(speaker_set, output_path):
    speakers = sorted(list(speaker_set))
    with open(output_path, 'w') as f:
        for i, spk in enumerate(speakers):
            f.write(spk + ' ' + str(i) + '\n')


def process_sentence(config: Dict[str, Any],
                     fp: Path,
                     sentences: Dict,
                     output_dir: Path,
                     mel_extractor=None):
    utt_id = fp.stem
    record = None
    if utt_id in sentences:
        # reading, resampling may occur
        wav, _ = librosa.load(str(fp), sr=config.fs)
        if len(wav.shape) != 1 or np.abs(wav).max() > 1.0:
            return record
        assert len(wav.shape) == 1, f"{utt_id} is not a mono-channel audio."
        assert np.abs(wav).max(
        ) <= 1.0, f"{utt_id} is seems to be different that 16 bit PCM."
        phones = sentences[utt_id][0]
        speaker = sentences[utt_id][1]
        logmel = mel_extractor.get_log_mel_fbank(wav, base='e')
        # change duration according to mel_length
        num_frames = logmel.shape[0]
        mel_dir = output_dir / "data_speech"
        mel_dir.mkdir(parents=True, exist_ok=True)
        mel_path = mel_dir / (utt_id + "_speech.npy")
        np.save(mel_path, logmel)
        record = {
            "utt_id": utt_id,
            "phones": phones,
            "text_lengths": len(phones),
            "speech_lengths": num_frames,
            "speech": str(mel_path),
            "speaker": speaker
        }
    return record


def process_sentences(config,
                      fps: List[Path],
                      sentences: Dict,
                      output_dir: Path,
                      mel_extractor=None,
                      nprocs: int=1):

    if nprocs == 1:
        results = []
        for fp in tqdm.tqdm(fps, total=len(fps)):
            record = process_sentence(
                config=config,
                fp=fp,
                sentences=sentences,
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
                                         sentences, output_dir, mel_extractor)
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
        default="ljspeech",
        type=str,
        help="name of dataset, should in {ljspeech} now")

    parser.add_argument(
        "--rootdir", default=None, type=str, help="directory to dataset.")

    parser.add_argument(
        "--dumpdir",
        type=str,
        required=True,
        help="directory to dump feature files.")

    parser.add_argument(
        "--config-path",
        default="conf/default.yaml",
        type=str,
        help="yaml format configuration file.")

    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)")
    parser.add_argument(
        "--num-cpu", type=int, default=1, help="number of process.")

    args = parser.parse_args()

    config_path = Path(args.config_path).resolve()
    root_dir = Path(args.rootdir).expanduser()
    dumpdir = Path(args.dumpdir).expanduser()
    # use absolute path
    dumpdir = dumpdir.resolve()
    dumpdir.mkdir(parents=True, exist_ok=True)

    assert root_dir.is_dir()

    with open(config_path, 'rt') as f:
        _C = yaml.safe_load(f)
        _C = Configuration(_C)
        config = _C.clone()

    if args.verbose > 1:
        print(vars(args))
        print(config)

    phone_id_map_path = dumpdir / "phone_id_map.txt"
    speaker_id_map_path = dumpdir / "speaker_id_map.txt"

    if args.dataset == "ljspeech":
        wav_files = sorted(list((root_dir / "wavs").rglob("*.wav")))
        frontend = English()
        sentences, speaker_set = get_lj_sentences(root_dir / "metadata.csv",
                                                  frontend)
        get_input_token(sentences, phone_id_map_path)
        get_spk_id_map(speaker_set, speaker_id_map_path)
        # split data into 3 sections
        num_train = 12900
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
            nprocs=args.num_cpu)
    if dev_wav_files:
        process_sentences(
            config=config,
            fps=dev_wav_files,
            sentences=sentences,
            output_dir=dev_dump_dir,
            mel_extractor=mel_extractor,
            nprocs=args.num_cpu)
    if test_wav_files:
        process_sentences(
            config=config,
            fps=test_wav_files,
            sentences=sentences,
            output_dir=test_dump_dir,
            mel_extractor=mel_extractor,
            nprocs=args.num_cpu)


if __name__ == "__main__":
    main()
