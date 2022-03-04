#!/usr/bin/python
import argparse
import os

import yaml


def change_speech_yaml(yaml_name: str, device: str):
    """Change the settings of the device under the voice task configuration file

    Args:
        yaml_name (str): asr or asr_pd or tts or tts_pd
        cpu (bool): True means set device to "cpu"
        model_type (dict): change model type
    """
    if "asr" in yaml_name:
        dirpath = "./conf/asr/"
    elif 'tts' in yaml_name:
        dirpath = "./conf/tts/"
    yamlfile = dirpath + yaml_name + ".yaml"
    tmp_yamlfile = dirpath + yaml_name + "_tmp.yaml"
    os.system("cp %s %s" % (yamlfile, tmp_yamlfile))

    with open(tmp_yamlfile) as f, open(yamlfile, "w+", encoding="utf-8") as fw:
        y = yaml.safe_load(f)
        if device == 'cpu':
            print("Set device: cpu")
            if yaml_name == 'asr':
                y['device'] = 'cpu'
            elif yaml_name == 'asr_pd':
                y['am_predictor_conf']['device'] = 'cpu'
            elif yaml_name == 'tts':
                y['device'] = 'cpu'
            elif yaml_name == 'tts_pd':
                y['am_predictor_conf']['device'] = 'cpu'
                y['voc_predictor_conf']['device'] = 'cpu'
        elif device == 'gpu':
            print("Set device: gpu")
            if yaml_name == 'asr':
                y['device'] = 'gpu:0'
            elif yaml_name == 'asr_pd':
                y['am_predictor_conf']['device'] = 'gpu:0'
            elif yaml_name == 'tts':
                y['device'] = 'gpu:0'
            elif yaml_name == 'tts_pd':
                y['am_predictor_conf']['device'] = 'gpu:0'
                y['voc_predictor_conf']['device'] = 'gpu:0'
        else:
            print("Please set correct device: cpu or gpu.")

        print("The content of '%s': " % (yamlfile))
        print(yaml.dump(y, default_flow_style=False, sort_keys=False))
        yaml.dump(y, fw, allow_unicode=True)
    os.system("rm %s" % (tmp_yamlfile))
    print("Change %s successfully." % (yamlfile))


def change_app_yaml(task: str, engine_type: str):
    """Change the engine type and corresponding configuration file of the speech task in application.yaml

    Args:
        task (str):  asr or tts
    """
    yamlfile = "./conf/application.yaml"
    tmp_yamlfile = "./conf/application_tmp.yaml"
    os.system("cp %s %s" % (yamlfile, tmp_yamlfile))
    with open(tmp_yamlfile) as f, open(yamlfile, "w+", encoding="utf-8") as fw:
        y = yaml.safe_load(f)
        y['engine_type'][task] = engine_type
        path_list = ["./conf/", task, "/", task]
        if engine_type == 'python':
            path_list.append(".yaml")

        elif engine_type == 'inference':
            path_list.append("_pd.yaml")
        y['engine_backend'][task] = ''.join(path_list)
        print("The content of './conf/application.yaml': ")
        print(yaml.dump(y, default_flow_style=False, sort_keys=False))
        yaml.dump(y, fw, allow_unicode=True)
    os.system("rm %s" % (tmp_yamlfile))
    print("Change %s successfully." % (yamlfile))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--change_task',
        type=str,
        default=None,
        help='Change task',
        choices=[
            'app-asr-python',
            'app-asr-inference',
            'app-tts-python',
            'app-tts-inference',
            'speech-asr-cpu',
            'speech-asr-gpu',
            'speech-asr_pd-cpu',
            'speech-asr_pd-gpu',
            'speech-tts-cpu',
            'speech-tts-gpu',
            'speech-tts_pd-cpu',
            'speech-tts_pd-gpu',
        ],
        required=True)
    args = parser.parse_args()

    types = args.change_task.split("-")
    if types[0] == "app":
        change_app_yaml(types[1], types[2])
    elif types[0] == "speech":
        change_speech_yaml(types[1], types[2])
    else:
        print("Error change task, please check change_task.")
