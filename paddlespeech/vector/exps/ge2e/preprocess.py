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
from pathlib import Path

from paddlespeech.vector.exps.ge2e.audio_processor import SpeakerVerificationPreprocessor
from paddlespeech.vector.exps.ge2e.config import get_cfg_defaults
from paddlespeech.vector.exps.ge2e.dataset_processors import process_aidatatang_200zh
from paddlespeech.vector.exps.ge2e.dataset_processors import process_librispeech
from paddlespeech.vector.exps.ge2e.dataset_processors import process_magicdata
from paddlespeech.vector.exps.ge2e.dataset_processors import process_voxceleb1
from paddlespeech.vector.exps.ge2e.dataset_processors import process_voxceleb2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="preprocess dataset for speaker verification task")
    parser.add_argument(
        "--datasets_root",
        type=Path,
        help="Path to the directory containing your LibriSpeech, LibriTTS and VoxCeleb datasets."
    )
    parser.add_argument(
        "--output_dir", type=Path, help="Path to save processed dataset.")
    parser.add_argument(
        "--dataset_names",
        type=str,
        default="librispeech_other,voxceleb1,voxceleb2",
        help="comma-separated list of names of the datasets you want to preprocess. only "
        "the train set of these datastes will be used. Possible names: librispeech_other, "
        "voxceleb1, voxceleb2, aidatatang_200zh, magicdata.")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Whether to skip output files with the same name. Useful if this script was interrupted."
    )
    parser.add_argument(
        "--no_trim",
        action="store_true",
        help="Preprocess audio without trimming silences (not recommended).")

    args = parser.parse_args()

    if not args.no_trim:
        try:
            import webrtcvad
            print(webrtcvad.__version__)
        except Exception as e:
            raise ModuleNotFoundError(
                "Package 'webrtcvad' not found. This package enables "
                "noise removal and is recommended. Please install and "
                "try again. If installation fails, "
                "use --no_trim to disable this error message.")
    del args.no_trim

    args.datasets = [item.strip() for item in args.dataset_names.split(",")]
    if not hasattr(args, "output_dir"):
        args.output_dir = args.dataset_root / "SV2TTS" / "encoder"

    args.output_dir = args.output_dir.expanduser()
    args.datasets_root = args.datasets_root.expanduser()
    assert args.datasets_root.exists()
    args.output_dir.mkdir(exist_ok=True, parents=True)

    config = get_cfg_defaults()
    print(args)

    c = config.data
    processor = SpeakerVerificationPreprocessor(
        sampling_rate=c.sampling_rate,
        audio_norm_target_dBFS=c.audio_norm_target_dBFS,
        vad_window_length=c.vad_window_length,
        vad_moving_average_width=c.vad_moving_average_width,
        vad_max_silence_length=c.vad_max_silence_length,
        mel_window_length=c.mel_window_length,
        mel_window_step=c.mel_window_step,
        n_mels=c.n_mels,
        partial_n_frames=c.partial_n_frames,
        min_pad_coverage=c.min_pad_coverage,
        partial_overlap_ratio=c.min_pad_coverage, )

    preprocess_func = {
        "librispeech_other": process_librispeech,
        "voxceleb1": process_voxceleb1,
        "voxceleb2": process_voxceleb2,
        "aidatatang_200zh": process_aidatatang_200zh,
        "magicdata": process_magicdata,
    }

    for dataset in args.datasets:
        print("Preprocessing %s" % dataset)
        preprocess_func[dataset](processor, args.datasets_root, args.output_dir,
                                 args.skip_existing)
