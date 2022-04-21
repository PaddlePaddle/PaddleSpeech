#!/usr/bin/python
import argparse
import os

import yaml
"""
def change_value1(yamlfile: str, target_key: str, target_value: str, engine: str="tts_online"):
    tmp_yamlfile = yamlfile.split(".yaml")[0] + "_tmp.yaml"
    os.system("cp %s %s" % (yamlfile, tmp_yamlfile))

    with open(tmp_yamlfile) as f, open(yamlfile, "w+", encoding="utf-8") as fw:
        y = yaml.safe_load(f)
        y[engine][target_key] = target_value

        print(yaml.dump(y, default_flow_style=False, sort_keys=False))
        yaml.dump(y, fw, allow_unicode=True)
    os.system("rm %s" % (tmp_yamlfile))
    print(f"Change key: {target_key} to value: {target_value} successfully.")

def change_protocol(yamlfile: str, target_key: str, target_value: str):
    tmp_yamlfile = yamlfile.split(".yaml")[0] + "_tmp.yaml"
    os.system("cp %s %s" % (yamlfile, tmp_yamlfile))

    with open(tmp_yamlfile) as f, open(yamlfile, "w+", encoding="utf-8") as fw:
        y = yaml.safe_load(f)
        y[target_key] = target_value

        print(yaml.dump(y, default_flow_style=False, sort_keys=False))
        yaml.dump(y, fw, allow_unicode=True)
    os.system("rm %s" % (tmp_yamlfile))
    print(f"Change key: {target_key} to value: {target_value} successfully.")

def change_engine_type(yamlfile: str, target_key: str, target_value: str):
    tmp_yamlfile = yamlfile.split(".yaml")[0] + "_tmp.yaml"
    os.system("cp %s %s" % (yamlfile, tmp_yamlfile))

    with open(tmp_yamlfile) as f, open(yamlfile, "w+", encoding="utf-8") as fw:
        y = yaml.safe_load(f)
        y[target_key] = [target_value]

        print(yaml.dump(y, default_flow_style=False, sort_keys=False))
        yaml.dump(y, fw, allow_unicode=True)
    os.system("rm %s" % (tmp_yamlfile))
    print(f"Change key: {target_key} to value: {target_value} successfully.")
"""


def change_value(args):
    yamlfile = args.config_file
    change_type = args.change_type
    engine_type = args.engine_type
    target_key = args.target_key
    target_value = args.target_value

    tmp_yamlfile = yamlfile.split(".yaml")[0] + "_tmp.yaml"
    os.system("cp %s %s" % (yamlfile, tmp_yamlfile))

    with open(tmp_yamlfile) as f, open(yamlfile, "w+", encoding="utf-8") as fw:
        y = yaml.safe_load(f)

        if change_type == "model":
            if engine_type == "tts_online-onnx":
                target_value = target_value + "_onnx"
            y[engine_type][target_key] = target_value
        elif change_type == "protocol":
            assert (target_key == "protocol" and (
                target_value == "http" or target_value == "websocket"
            )), "if change_type is protocol, target_key must be set protocol."
            y[target_key] = target_value
        elif change_type == "engine_type":
            assert (
                target_key == "engine_list"
            ), "if change_type is engine_type, target_key must be set engine_list."
            y[target_key] = [target_value]
        elif change_type == "device":
            assert (
                target_key == "device"
            ), "if change_type is device, target_key must be set device."
            if y["engine_list"][0] == "tts_online":
                y["tts_online"]["device"] = target_value
            elif y["engine_list"][0] == "tts_online-onnx":
                y["tts_online-onnx"]["am_sess_conf"]["device"] = target_value
                y["tts_online-onnx"]["voc_sess_conf"]["device"] = target_value
            else:
                print(
                    "Error engine_list, please set tts_online or tts_online-onnx"
                )

        else:
            print("Error change_type, please set correct change_type.")

        print(yaml.dump(y, default_flow_style=False, sort_keys=False))
        yaml.dump(y, fw, allow_unicode=True)
    os.system("rm %s" % (tmp_yamlfile))
    print(f"Change key: {target_key} to value: {target_value} successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_file',
        type=str,
        default='./conf/application.yaml',
        help='server yaml file.')
    parser.add_argument(
        '--change_type',
        type=str,
        default="model",
        choices=["model", "protocol", "engine_type", "device"],
        help='change protocol', )
    parser.add_argument(
        '--engine_type',
        type=str,
        default="tts_online",
        help='engine type',
        choices=["tts_online", "tts_online-onnx"])
    parser.add_argument(
        '--target_key',
        type=str,
        default=None,
        help='Change key',
        required=True)
    parser.add_argument(
        '--target_value',
        type=str,
        default=None,
        help='target value',
        required=True)

    args = parser.parse_args()

    change_value(args)
    """
    if args.change_type == "model":
        change_value(args.config_file, args.target_key, args.target_value, args.engine)
    elif args.change_type == "protocol":
        change_protocol(args.config_file, args.target_key, args.target_value)
    else:
        print("Please set correct change type, model or protocol")
    """
