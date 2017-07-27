"""
This tool is used for preparing data for DeepSpeech2 trainning on paddle cloud.

Steps:
1. Read original manifest and get the local path of sound files.
2. Tar all local sound files into one tar file.
3. Modify original manifest to remove the local path information.

Finally, we will get a tar file and a manifest with sound file name, duration
and text.
"""
import json
import os
import tarfile
import sys
import argparse
sys.path.append('../')
from data_utils.utils import read_manifest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--manifest_path",
    default="../datasets/manifest.train",
    type=str,
    help="Manifest of target data. (default: %(default)s)")
parser.add_argument(
    "--out_tar_path",
    default="./data/dev.tar",
    type=str,
    help="Output tar file path. (default: %(default)s)")
parser.add_argument(
    "--out_manifest_path",
    default="./data/dev.mani",
    type=str,
    help="Manifest of output data. (default: %(default)s)")
args = parser.parse_args()


def gen_pcloud_data(manifest_path, out_tar_path, out_manifest_path):
    '''
    1. According manifest, tar sound files into out_tar_path
    2. Generate a new manifest for output tar file
    '''
    out_tar = tarfile.open(out_tar_path, 'w')
    manifest = read_manifest(manifest_path)
    results = []
    for json_data in manifest:
        sound_file = json_data['audio_filepath']
        filename = os.path.basename(sound_file)
        out_tar.add(sound_file, arcname=filename)
        json_data['audio_filepath'] = filename
        results.append("%s\n" % json.dumps(json_data))
    with open(out_manifest_path, 'w') as out_manifest:
        out_manifest.writelines(results)
    out_manifest.close()
    out_tar.close()


if __name__ == '__main__':
    gen_pcloud_data(args.manifest_path, args.out_tar_path,
                    args.out_manifest_path)
