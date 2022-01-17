#!/usr/bin/python3
#! coding:utf-8

import cProfile
import argparse
import yaml
from yacs.config import CfgNode
from paddlespeech.s2t.training.cli import default_argument_parser
from paddlespeech.sid.bin.trainer import SIDTrainer as Trainer
import paddle
import os
def main_sp(config, args):
    exp = Trainer(config, args)
    exp.setup()
    exp.run()
    
def main():
    # parse args and config and redirect to train_sp

    parser = argparse.ArgumentParser(description="Train a ECAPA-XVECTOR model.")
    parser = default_argument_parser(parser)

    args = parser.parse_args()
    print(args)
    config = CfgNode(new_allowed=True)
    if args.config:
        config.merge_from_file(args.config)
    config.freeze()
    print(config)

    if args.dump_config:
        print("dump the config to: {}".format(args.dump_config))
        with open(args.dump_config, 'w') as f:
            print(config, file=f)

    
    # pr = cProfile.Profile()
    # pr.runcall(main, config, args)
    # pr.dump_stats()
    if args.ngpu <= 0:
        paddle.device.set_device("cpu")
    main_sp(config, args)
if __name__ == "__main__":
    main()
