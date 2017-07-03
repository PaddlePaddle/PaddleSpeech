import os
import json
import argparse


def split_data(inManifest, tar_path, outManifest):
    trainer_id = 1
    trainer_count = 2
    #with open("/trainer_id", "r") as f:
    #    trainer_id = int(f.readline()[:-1])
    #with open("/trainer_count", "r") as f:
    #    trainer_count = int(f.readline()[:-1])

    tarPath = os.path.abspath(tar_path)
    result = []
    for index, json_line in enumerate(open(inManifest)):
        if (index % trainer_count) == trainer_id:
            json_data = json.loads(json_line)
            json_data['audio_filepath'] = "tar:%s#%s" % (
                tarPath, json_data['audio_filepath'])
            result.append("%s\n" % json.dumps(json_data))
    with open(outManifest, 'w') as manifest:
        manifest.writelines(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--in_manifest_path",
        default='datasets/dev.mani',
        type=str,
        help="Input manifest path. (default: %(default)s)")
    parser.add_argument(
        "--data_tar_path",
        default='datasets/dev.tar',
        type=str,
        help="Data tar file path. (default: %(default)s)")
    parser.add_argument(
        "--out_manifest_path",
        default='datasets/dev.mani.split',
        type=str,
        help="Out manifest file path. (default: %(default)s)")
    args = parser.parse_args()

    split_data(args.in_manifest_path, args.data_tar_path,
               args.out_manifest_path)
