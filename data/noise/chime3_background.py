"""Prepare CHiME3 background data.

Download, unpack and create manifest files.
Manifest file is a json-format file with each line containing the
meta data (i.e. audio filepath, transcript and audio duration)
of each audio file in the data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import distutils.util
import os
import wget
import zipfile
import argparse
import soundfile
import json
from paddle.v2.dataset.common import md5file

DATA_HOME = os.path.expanduser('~/.cache/paddle/dataset/speech')

URL = "https://d4s.myairbridge.com/packagev2/AG0Y3DNBE5IWRRTV/?dlid=W19XG7T0NNHB027139H0EQ"
MD5 = "c3ff512618d7a67d4f85566ea1bc39ec"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--target_dir",
    default=DATA_HOME + "/chime3_background",
    type=str,
    help="Directory to save the dataset. (default: %(default)s)")
parser.add_argument(
    "--manifest_filepath",
    default="manifest.chime3.background",
    type=str,
    help="Filepath for output manifests. (default: %(default)s)")
args = parser.parse_args()


def download(url, md5sum, target_dir, filename=None):
    """Download file from url to target_dir, and check md5sum."""
    if filename == None:
        filename = url.split("/")[-1]
    if not os.path.exists(target_dir): os.makedirs(target_dir)
    filepath = os.path.join(target_dir, filename)
    if not (os.path.exists(filepath) and md5file(filepath) == md5sum):
        print("Downloading %s ..." % url)
        wget.download(url, target_dir)
        print("\nMD5 Chesksum %s ..." % filepath)
        if not md5file(filepath) == md5sum:
            raise RuntimeError("MD5 checksum failed.")
    else:
        print("File exists, skip downloading. (%s)" % filepath)
    return filepath


def unpack(filepath, target_dir):
    """Unpack the file to the target_dir."""
    print("Unpacking %s ..." % filepath)
    if filepath.endswith('.zip'):
        zip = zipfile.ZipFile(filepath, 'r')
        zip.extractall(target_dir)
        zip.close()
    elif filepath.endswith('.tar') or filepath.endswith('.tar.gz'):
        tar = zipfile.open(filepath)
        tar.extractall(target_dir)
        tar.close()
    else:
        raise ValueError("File format is not supported for unpacking.")


def create_manifest(data_dir, manifest_path):
    """Create a manifest json file summarizing the data set, with each line
    containing the meta data (i.e. audio filepath, transcription text, audio
    duration) of each audio file within the data set.
    """
    print("Creating manifest %s ..." % manifest_path)
    json_lines = []
    for subfolder, _, filelist in sorted(os.walk(data_dir)):
        for filename in filelist:
            if filename.endswith('.wav'):
                filepath = os.path.join(data_dir, subfolder, filename)
                audio_data, samplerate = soundfile.read(filepath)
                duration = float(len(audio_data)) / samplerate
                json_lines.append(
                    json.dumps({
                        'audio_filepath': filepath,
                        'duration': duration,
                        'text': ''
                    }))
    with open(manifest_path, 'w') as out_file:
        for line in json_lines:
            out_file.write(line + '\n')


def prepare_chime3(url, md5sum, target_dir, manifest_path):
    """Download, unpack and create summmary manifest file."""
    if not os.path.exists(os.path.join(target_dir, "CHiME3")):
        # download
        filepath = download(url, md5sum, target_dir,
                            "myairbridge-AG0Y3DNBE5IWRRTV.zip")
        # unpack
        unpack(filepath, target_dir)
        unpack(
            os.path.join(target_dir, 'CHiME3_background_bus.zip'), target_dir)
        unpack(
            os.path.join(target_dir, 'CHiME3_background_caf.zip'), target_dir)
        unpack(
            os.path.join(target_dir, 'CHiME3_background_ped.zip'), target_dir)
        unpack(
            os.path.join(target_dir, 'CHiME3_background_str.zip'), target_dir)
    else:
        print("Skip downloading and unpacking. Data already exists in %s." %
              target_dir)
    # create manifest json file
    create_manifest(target_dir, manifest_path)


def main():
    prepare_chime3(
        url=URL,
        md5sum=MD5,
        target_dir=args.target_dir,
        manifest_path=args.manifest_filepath)


if __name__ == '__main__':
    main()
