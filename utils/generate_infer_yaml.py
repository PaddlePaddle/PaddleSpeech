#!/usr/bin/env python3
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
'''
    Merge training configs into a single inference config.
    The single inference config is for CLI, which only takes a single config to do inferencing.
    The trainig configs includes: model config, preprocess config, decode config, vocab file and cmvn file.

    Process:
    # step 1: prepare dir
    mkdir -p release_dir
    cp -r exp conf data release_dir
    cd release_dir 
 
    # step 2: get "model.yaml" which conatains all configuration info.
    # if does not contain preprocess.yaml file. e.g ds2:
    python generate_infer_yaml.py --cfg_pth conf/deepspeech2_online.yaml --dcd_pth conf/tuning/chunk_decode.yaml --vb_pth data/lang_char/vocab.txt --cmvn_pth data/mean_std.json --save_pth model.yaml --pre_pth null        
    # if contains preprocess.yaml file. e.g  u2:
    python generate_infer_yaml.py --cfg_pth conf/chunk_conformer.yaml --dcd_pth conf/tuning/chunk_decode.yaml --vb_pth data/lang_char/vocab.txt --cmvn_pth data/mean_std.json --save_pth model.yaml --pre_pth conf/preprocess.yaml
 
    # step 3:  remove redundant things
    rm xxx       

    # step 4: tar file
    # ds2
    tar czvf asr0_deepspeech2_online_aishell_ckpt_0.2.0.model.tar.gz model.yaml conf data/ exp/
    # u2
    tar czvf asr1_chunk_conformer_aishell_ckpt_0.2.0.model.tar.gz model.yaml conf data/ exp/  
'''
import argparse
import json
import math
import os
from contextlib import redirect_stdout

from yacs.config import CfgNode

from paddlespeech.s2t.frontend.utility import load_dict


def save(save_path, config):
    with open(save_path, 'w') as fp:
        with redirect_stdout(fp):
            print(config.dump())


def load(save_path):
    config = CfgNode(new_allowed=True)
    config.merge_from_file(save_path)
    return config


def load_json(json_path):
    with open(json_path) as f:
        json_content = json.load(f)
    return json_content


def remove_config_part(config, key_list):
    if len(key_list) == 0:
        return
    for i in range(len(key_list) - 1):
        config = config[key_list[i]]
    config.pop(key_list[-1])


def load_cmvn_from_json(cmvn_stats):
    means = cmvn_stats['mean_stat']
    variance = cmvn_stats['var_stat']
    count = cmvn_stats['frame_num']
    for i in range(len(means)):
        means[i] /= count
        variance[i] = variance[i] / count - means[i] * means[i]
        if variance[i] < 1.0e-20:
            variance[i] = 1.0e-20
        variance[i] = 1.0 / math.sqrt(variance[i])
    cmvn_stats = {"mean": means, "istd": variance}
    return cmvn_stats


def merge_configs(
        conf_path="conf/conformer.yaml",
        preprocess_path="conf/preprocess.yaml",
        decode_path="conf/tuning/decode.yaml",
        vocab_path="data/vocab.txt",
        cmvn_path="data/mean_std.json",
        save_path="conf/conformer_infer.yaml", ):

    # Load the configs
    config = load(conf_path)
    decode_config = load(decode_path)
    vocab_list = load_dict(vocab_path)

    # If use the kaldi feature, do not load the cmvn file
    if cmvn_path.split(".")[-1] == 'json':
        cmvn_stats = load_json(cmvn_path)
        if os.path.exists(preprocess_path):
            preprocess_config = load(preprocess_path)
            for idx, process in enumerate(preprocess_config["process"]):
                if process['type'] == "cmvn_json":
                    preprocess_config["process"][idx]["cmvn_path"] = cmvn_stats
                    break

            config.preprocess_config = preprocess_config
        else:
            cmvn_stats = load_cmvn_from_json(cmvn_stats)
            config.mean_std_filepath = [{"cmvn_stats": cmvn_stats}]
            config.augmentation_config = ''
    # the cmvn file is end with .ark
    else:
        config.cmvn_path = cmvn_path
    # Updata the config
    config.vocab_filepath = vocab_list
    config.input_dim = config.feat_dim
    config.output_dim = len(config.vocab_filepath)
    config.decode = decode_config
    # Remove some parts of the config

    if os.path.exists(preprocess_path):
        remove_train_list = [
            "train_manifest",
            "dev_manifest",
            "test_manifest",
            "n_epoch",
            "accum_grad",
            "global_grad_clip",
            "optim",
            "optim_conf",
            "scheduler",
            "scheduler_conf",
            "log_interval",
            "checkpoint",
            "shuffle_method",
            "weight_decay",
            "ctc_grad_norm_type",
            "minibatches",
            "subsampling_factor",
            "batch_bins",
            "batch_count",
            "batch_frames_in",
            "batch_frames_inout",
            "batch_frames_out",
            "sortagrad",
            "feat_dim",
            "stride_ms",
            "window_ms",
            "batch_size",
            "maxlen_in",
            "maxlen_out",
        ]
    else:
        remove_train_list = [
            "train_manifest",
            "dev_manifest",
            "test_manifest",
            "n_epoch",
            "accum_grad",
            "global_grad_clip",
            "log_interval",
            "checkpoint",
            "lr",
            "lr_decay",
            "batch_size",
            "shuffle_method",
            "weight_decay",
            "sortagrad",
            "num_workers",
        ]

    for item in remove_train_list:
        try:
            remove_config_part(config, [item])
        except Exception as e:
            print(item + " " + "can not be removed")

    # Save the config
    save(save_path, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Config merge', add_help=True)
    parser.add_argument(
        '--cfg_pth',
        type=str,
        default='conf/transformer.yaml',
        help='origin config file')
    parser.add_argument(
        '--pre_pth', type=str, default="conf/preprocess.yaml", help='')
    parser.add_argument(
        '--dcd_pth', type=str, default="conf/tuninig/decode.yaml", help='')
    parser.add_argument(
        '--vb_pth', type=str, default="data/lang_char/vocab.txt", help='')
    parser.add_argument(
        '--cmvn_pth', type=str, default="data/mean_std.json", help='')
    parser.add_argument(
        '--save_pth', type=str, default="conf/transformer_infer.yaml", help='')
    parser_args = parser.parse_args()

    merge_configs(
        conf_path=parser_args.cfg_pth,
        decode_path=parser_args.dcd_pth,
        preprocess_path=parser_args.pre_pth,
        vocab_path=parser_args.vb_pth,
        cmvn_path=parser_args.cmvn_pth,
        save_path=parser_args.save_pth, )
