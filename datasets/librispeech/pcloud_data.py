import json
import os
import tarfile
import sys
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--manifest_path",
    default="/manifest.train",
    type=str,
    help="Manifest of target data. (default: %(default)s)")
parser.add_argument(
    "--out_tar_path",
    default="/dev.tar",
    type=str,
    help="Output tar file path. (default: %(default)s)")
parser.add_argument(
    "--out_manifest_path",
    default="/dev.mani",
    type=str,
    help="Manifest of output data. (default: %(default)s)")
args = parser.parse_args()


def gen_pcloud_data(manifest_path, out_tar_path, out_manifest_path):
    '''
    1. According manifest, tar sound files into out_tar_path
    2. Generate a new manifest for output tar file
    '''
    out_tar = tarfile.open(out_tar_path, 'w')
    manifest = []
    for json_line in open(manifest_path):
        try:
            json_data = json.loads(json_line)
        except Exception as e:
            raise IOError("Error reading manifest: %s" % str(e))
        sound_file = json_data['audio_filepath']
        filename = os.path.basename(sound_file)
        out_tar.add(sound_file, arcname=filename)
        json_data['audio_filepath'] = filename
        manifest.append("%s\n" % json.dumps(json_data))
    with open(out_manifest_path, 'w') as out_manifest:
        out_manifest.writelines(manifest)
    out_manifest.close()
    out_tar.close()


if __name__ == '__main__':
    gen_pcloud_data(args.manifest_path, args.out_tar_path,
                    args.out_manifest_path)
