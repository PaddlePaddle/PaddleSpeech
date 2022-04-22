# #!/bin/bash
import os

now_dir = os.path.abspath("./")
print(now_dir)
MAIN_ROOT = './../../../`'
UTILS_PATH = MAIN_ROOT + "utils/"
LC_ALL = "C"
PYTHONDONTWRITEBYTECODE = 1
PYTHONIOENCODING = "UTF-8"
PYTHONPATH = MAIN_ROOT
MODEL = "fastspeech2"
BIN_DIR = MAIN_ROOT + f"/paddlespeech/t2s/exps/{MODEL}"

gpus=0
stage=0
stop_stage=100

conf_path="conf/default.yaml"
train_output_path="exp/default"
ckpt_name="snapshot_iter_153.pdz"
