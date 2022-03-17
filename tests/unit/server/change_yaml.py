#!/usr/bin/python
import argparse
import os

import yaml


def change_device(yamlfile: str, engine: str, device: str):
    """Change the settings of the device under the voice task configuration file

    Args:
        yaml_name (str): asr or asr_pd or tts or tts_pd
        cpu (bool): True means set device to "cpu"
        model_type (dict): change model type
    """
    tmp_yamlfile = yamlfile.split(".yaml")[0] + "_tmp.yaml"
    os.system("cp %s %s" % (yamlfile, tmp_yamlfile))

    if device == 'cpu':
        set_device = 'cpu'
    elif device == 'gpu':
        set_device = 'gpu:0'
    else:
        print("Please set correct device: cpu or gpu.")

    with open(tmp_yamlfile) as f, open(yamlfile, "w+", encoding="utf-8") as fw:
        y = yaml.safe_load(f)
        if engine == 'asr_python' or engine == 'tts_python' or engine == 'cls_python':
            y[engine]['device'] = set_device
        elif engine == 'asr_inference':
            y[engine]['am_predictor_conf']['device'] = set_device
        elif engine == 'tts_inference':
            y[engine]['am_predictor_conf']['device'] = set_device
            y[engine]['voc_predictor_conf']['device'] = set_device
        elif engine == 'cls_inference':
            y[engine]['predictor_conf']['device'] = set_device
        else:
            print(
                "Please set correct engine: asr_python, tts_python, asr_inference, tts_inference."
            )

        print(yaml.dump(y, default_flow_style=False, sort_keys=False))
        yaml.dump(y, fw, allow_unicode=True)
    os.system("rm %s" % (tmp_yamlfile))
    print("Change %s successfully." % (yamlfile))


def change_engine_type(yamlfile: str, engine_type):
    """Change the engine type and corresponding configuration file of the speech task in application.yaml

    Args:
        task (str):  asr or tts
    """
    tmp_yamlfile = yamlfile.split(".yaml")[0] + "_tmp.yaml"
    os.system("cp %s %s" % (yamlfile, tmp_yamlfile))
    speech_task = engine_type.split("_")[0]

    with open(tmp_yamlfile) as f, open(yamlfile, "w+", encoding="utf-8") as fw:
        y = yaml.safe_load(f)
        engine_list = y['engine_list']
        for engine in engine_list:
            if speech_task in engine:
                engine_list.remove(engine)
                engine_list.append(engine_type)
        y['engine_list'] = engine_list
        print(yaml.dump(y, default_flow_style=False, sort_keys=False))
        yaml.dump(y, fw, allow_unicode=True)
    os.system("rm %s" % (tmp_yamlfile))
    print("Change %s successfully." % (yamlfile))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_file',
        type=str,
        default='./conf/application.yaml',
        help='server yaml file.')
    parser.add_argument(
        '--change_task',
        type=str,
        default=None,
        help='Change task',
        choices=[
            'enginetype-asr_python',
            'enginetype-asr_inference',
            'enginetype-tts_python',
            'enginetype-tts_inference',
            'enginetype-cls_python',
            'enginetype-cls_inference',
            'device-asr_python-cpu',
            'device-asr_python-gpu',
            'device-asr_inference-cpu',
            'device-asr_inference-gpu',
            'device-tts_python-cpu',
            'device-tts_python-gpu',
            'device-tts_inference-cpu',
            'device-tts_inference-gpu',
            'device-cls_python-cpu',
            'device-cls_python-gpu',
            'device-cls_inference-cpu',
            'device-cls_inference-gpu',
        ],
        required=True)
    args = parser.parse_args()

    types = args.change_task.split("-")
    if types[0] == "enginetype":
        change_engine_type(args.config_file, types[1])
    elif types[0] == "device":
        change_device(args.config_file, types[1], types[2])
    else:
        print("Error change task, please check change_task.")
