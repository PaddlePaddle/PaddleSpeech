import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tqdm

from paddlespeech.cli.vector import VectorExecutor


def _process_utterance(ifpath: Path, input_dir: Path, output_dir: Path,
                       vec_executor):
    rel_path = ifpath.relative_to(input_dir)
    ofpath = (output_dir / rel_path).with_suffix(".npy")
    ofpath.parent.mkdir(parents=True, exist_ok=True)
    embed = vec_executor(audio_file=ifpath, force_yes=True)
    np.save(ofpath, embed)
    return ofpath


def main(args):
    # input output preparation
    input_dir = Path(args.input).expanduser()
    ifpaths = list(input_dir.rglob(args.pattern))
    print(f"{len(ifpaths)} utterances in total")
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    vec_executor = VectorExecutor()
    nprocs = args.num_cpu

    # warm up
    vec_executor(audio_file=ifpaths[0], force_yes=True)

    if nprocs == 1:
        results = []
        for ifpath in tqdm.tqdm(ifpaths, total=len(ifpaths)):
            _process_utterance(ifpath=ifpath,
                               input_dir=input_dir,
                               output_dir=output_dir,
                               vec_executor=vec_executor)
    else:
        with ThreadPoolExecutor(nprocs) as pool:
            with tqdm.tqdm(total=len(ifpaths)) as progress:
                for ifpath in ifpaths:
                    future = pool.submit(_process_utterance, ifpath, input_dir,
                                         output_dir, vec_executor)
                    future.add_done_callback(lambda p: progress.update())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute utterance embed.")
    parser.add_argument("--input",
                        type=str,
                        help="path of the audio_file folder.")
    parser.add_argument("--pattern",
                        type=str,
                        default="*.wav",
                        help="pattern to filter audio files.")
    parser.add_argument("--output",
                        metavar="OUTPUT_DIR",
                        help="path to save spk embedding results.")
    parser.add_argument("--num-cpu",
                        type=int,
                        default=1,
                        help="number of process.")
    args = parser.parse_args()

    main(args)
