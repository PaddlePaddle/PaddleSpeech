#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright     2019 ~ 2020    Zeng Xingui(zengxingui@baidu.com)
#
########################################################################

"""
Parameter management
"""
import os
import sys
import configparser
import logging
import argparse
import json
import torch

import sid_nnet_training.util.log as log


class ParamBase(object):
    """
    Base parameter manager
    """

    def __init__(self, val_dict):
        names = self.__dict__
        self.keys = []
        for key, value in val_dict.items():
            names[key] = value
            self.keys.append(key)

    def check(self):
        """
        check parameters
        """

        return True


class CommonBaseParam(ParamBase):
    """
    common base  parameter manager
    """

    dataset_type_list = ["kaldi_egs", "kaldi_feats"]
    dataloader_mode_list = ["train", "test"]

    def __init__(self, val_dict):
        super(CommonBaseParam, self).__init__(val_dict)
        self.format_params()

    def format_params(self):
        """
        Format the params
        """
        self.datasets = json.loads(self.datasets.strip().replace("'", '"'))
        self.print_freq = int(self.print_freq)
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if self.gpus.lower() == "all":
                self.gpus = list(range(0, device_count))
            else:
                self.gpus = [int(item) for item in self.gpus.split(',')]

    def check(self):
        """
        check the parameters
        """
        for val in self.datasets.values():
            if val not in CommonBaseParam.dataloader_mode_list:
                return False
        if self.dataset_type not in CommonBaseParam.dataset_type_list:
            return False

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            for gpu_id in self.gpus:
                if gpu_id >= device_count:
                    logger.error("Unvalid gpu id: %d" % (gpu_id))
                    return False

        return self.print_freq > 0

    @staticmethod
    def template():
        """
        Generate a string list of template
        """

        str_list = []
        str_list.append("[CommonBase]")
        str_list.append("# Dictionary. Dataset definition. key=dataset_name, value=dataloader_mode."
                        " name can be any string, value={%s}." % (",".join(CommonBaseParam.dataloader_mode_list)))
        str_list.append("# e.g.{'trainset': 'train', 'devset': 'test', 'testset': 'test'}")
        str_list.append("datasets = {'trainset': 'train', 'devset': 'test', 'testset':'test'}")
        str_list.append("# Type of dataset. value = {%s}" % (", ".join(CommonBaseParam.dataset_type_list)))
        str_list.append("dataset_type = kaldi_egs")
        str_list.append("# Gpus used to do the training. value = all or gpu id list(seperate with ','), e.g. 0,1,2 ")
        str_list.append("gpus = all")
        str_list.append("# int, print frequence.")
        str_list.append("print_freq = 10")

        return str_list


class ModelCommonParam(ParamBase):
    """
    ModelCommonParam parameter manager
    """
    model_type_list = ["xvector", "dvector", "att-xvector", "resnet18", "resnet34", "resnet50"]

    def __init__(self, val_dict):
        super(ModelCommonParam, self).__init__(val_dict)
        self.format_params()

    def format_params(self):
        """
        Format the params
        """
        self.nin = int(self.nin)
        self.nout = int(self.nout)

    def check(self):
        """
        check the parameters
        """
        if self.model_type not in ModelCommonParam.model_type_list:
            return False

        return True

    @staticmethod
    def template():
        """
        Generate a string list of template
        """

        str_list = []
        str_list.append("[ModelCommon]")
        str_list.append("# model type, value = {%s}" % (", ".join(ModelCommonParam.model_type_list)))
        str_list.append("model_type = xvector")
        str_list.append("# input dim of feature")
        str_list.append("nin = 23")
        str_list.append("# number of nodes of output layer")
        str_list.append("nout = 4608")

        return str_list


class TrainingParam(ParamBase):
    """
    Training parameter manager
    """
    loss_type_list = ["CE", "ASoftmax", "AMSoftmax", "FocalLoss", "GE2E", "MSLoss"]

    def __init__(self, val_dict):
        super(TrainingParam, self).__init__(val_dict)
        self.format_params()

    def parse_epoch_value_str(self, epoch_value_str):
        """
        Parse a epoch specific setting string

        Args:
            epoch_value_str: epoch specific setting string

        Returns:
            a list of setting value, length=max_epoch
        """
        value_list = []
        for item in epoch_value_str.strip().split(';'):
            epoch_str, value_str = item.strip().split('|')
            bg_epoch = end_epoch = -1
            if ':' in epoch_str:
                bg_epoch, end_epoch = [int(epoch) for epoch in epoch_str.strip().split(':')]
            else:
                bg_epoch = int(epoch_str.strip())
                end_epoch = bg_epoch + 1
            for epoch in range(bg_epoch, end_epoch):
                value_list.append(value_str)

        return value_list

    def parse_dropout_proportion(self, dropout_str):
        """
        Parse dropout proportion para from the config file

        Args:
            dropout_str: dropout setting string

        Returns:
            a list of dropout proportion
        """
        dropout_proportion_list = []
        for item in dropout_str.strip().split(';'):
            epoch_str, value_str = item.strip().split('|')
            bg_epoch = end_epoch = -1
            if ':' in epoch_str:
                bg_epoch, end_epoch = [int(epoch) for epoch in epoch_str.strip().split(':')]
            else:
                bg_epoch = int(epoch_str.strip())
                end_epoch = bg_epoch + 1
            if ':' in value_str:
                bg_dp, end_dp = [float(dp) for dp in value_str.strip().split(':')]
                for epoch in range(bg_epoch, end_epoch):
                    dp = (end_dp - bg_dp) * (epoch - bg_epoch) / (end_epoch - bg_epoch) + bg_dp
                    dropout_proportion_list.append(dp)
            else:
                for epoch in range(bg_epoch, end_epoch):
                    dropout_proportion_list.append(float(value_str))
        return dropout_proportion_list

    def format_params(self):
        """
        Format the params
        """
        self.lr = float(self.lr)
        self.asgd_parallel = [int(item) for item in self.parse_epoch_value_str(self.asgd_parallel)]
        self.asgd_selection = self.parse_epoch_value_str(self.asgd_selection)
        self.batch_size = [int(item) for item in self.parse_epoch_value_str(self.batch_size)]
        self.clip_grad_norm = [float(item) for item in self.parse_epoch_value_str(self.clip_grad_norm)]
        self.dropout_proportion = [float(item) for item in self.parse_dropout_proportion(self.dropout_proportion)]
        self.max_epoch = int(self.max_epoch)
        self.momentum = float(self.momentum)
        self.weight_decay = float(self.weight_decay)

    def check(self):
        """
        check the parameters

        Args:
            set_type: type of dataset, value = {train, dev, test}
        """
        assert len(self.asgd_parallel) == self.max_epoch
        assert len(self.asgd_selection) == self.max_epoch
        assert len(self.batch_size) == self.max_epoch
        assert len(self.clip_grad_norm) == self.max_epoch
        assert len(self.dropout_proportion) == self.max_epoch

        if self.loss_type not in TrainingParam.loss_type_list:
            return False

        return True

    @staticmethod
    def template():
        """
        Generate a string list of template
        """

        str_list = []
        str_list.append("[Training]")
        str_list.append("# loss funtion, value = {%s}" % (", ".join(TrainingParam.loss_type_list)))
        str_list.append("loss_type = CE")
        str_list.append("# learning rate")
        str_list.append("lr = 0.4")
        str_list.append("# ASGD setting, e.g. 0|3;1:4|4")
        str_list.append("asgd_parallel = 0|2")
        str_list.append("# ASGD selection policy, e.g. 0|best;1:3|average;3|clustering")
        str_list.append("asgd_selection =0|best")
        str_list.append("# mini batch size, e.g. 0|32;1:4|64")
        str_list.append("batch_size = 0|32")
        str_list.append("# max training epoch")
        str_list.append("max_epoch = 1")
        str_list.append("# optimizer, value = {SGD, Adam}, default = SGD")
        str_list.append("optimizer = SGD")
        str_list.append("# momentum for SGD, default = 0.5")
        str_list.append("momentum = 0.3")
        str_list.append("# weight_decay for SGD, default = 0.3")
        str_list.append("weight_decay = 0.00001")
        str_list.append("# clip grad norm, e.g. 0|1.5;1:4|2.0")
        str_list.append("clip_grad_norm = 0|1.5")
        str_list.append("# dropout proportion, e.g. 0|0;1:5|0.01;5:10|0.02:0.1")
        str_list.append("dropout_proportion = 0|0")

        return str_list


class FinetuneParam(ParamBase):
    """
    FinetuneParam parameter manager
    """
    def __init__(self, val_dict):
        super(FinetuneParam, self).__init__(val_dict)
        self.format_params()

    def format_params(self):
        """
        Format the params
        """
        self.bg_epoch = int(self.bg_epoch)
        if self.seed_model == "None":
            self.seed_model = None

    def check(self):
        """
        check parameters
        """
        if self.seed_model is not None:
            return os.path.exists(self.seed_model)
        return True

    @staticmethod
    def template():
        """
        Generate a string list of template
        """

        str_list = []
        str_list.append("[Finetune]")
        str_list.append("# the epoch index to start the training, default = 0")
        str_list.append("bg_epoch = 0")
        str_list.append("# seed model which used to initial the model parameters, default = None")
        str_list.append("seed_model = None")

        return str_list


# Dataset config
class KaldiEgsDatasetParam(ParamBase):
    """
    kaldi egs dataset parameter manager
    """
    def __init__(self, val_dict):
        super(KaldiEgsDatasetParam, self).__init__(val_dict)

    def get_setfile(self, set_name):
        """
        get the dataset file

        Args:
            set_name: name of dataset

        Returns:
            dataset files
        """
        names = self.__dict__
        key = set_name + "_file"

        return names[key]

    @staticmethod
    def template():
        """
        Generate a string list of template
        """

        str_list = []
        str_list.append("[KaldiEgsDataset]")
        str_list.append("# dataset files are defined at here, but the name should be consistant with"
                        " the value of datasets in CommonBase section")
        str_list.append("# e.g. for xvector training, 'trainset_file, devset_file, testset_file' must be defined")
        str_list.append("trainset_file = ")
        str_list.append("devset_file = ")
        str_list.append("testset_file = ")

        return str_list


class KaldiFeatsDatasetParam(ParamBase):
    """
    kaldi feats dataset parameter manager
    """
    def __init__(self, val_dict):
        super(KaldiFeatsDatasetParam, self).__init__(val_dict)
        self.format_params()

    def format_params(self):
        """
        Format the params
        """
        self.min_chunk_size = int(self.min_chunk_size)
        self.max_chunk_size = int(self.max_chunk_size)
        self.min_item_size = int(self.min_item_size)
        self.max_item_size = int(self.max_item_size)
        self.repeat = int(self.repeat)

    def check(self):
        """
        check parameters
        """
        assert self.min_chunk_size > 0 and self.min_chunk_size <= self.max_chunk_size
        assert self.repeat > 0

        return True

    def get_setfile(self, set_name):
        """
        get the dataset file

        Args:
            set_name: name of dataset

        Returns:
            dataset files
        """
        names = self.__dict__
        scp_key = set_name + "_feats_scp"
        int2utt_key = set_name + "_int2utt"

        return names[scp_key], names[int2utt_key]

    @staticmethod
    def template():
        """
        Generate a string list of template
        """

        str_list = []
        str_list.append("[KaldiFeatsDataset]")
        str_list.append("# dataset files are defined at here, but the name should be consistant with"
                        " the value of datasets in CommonBase section")
        str_list.append("# e.g. for xvector training, 'trainset_file, devset_file, testset_file' must be defined")
        str_list.append("trainset_feats_scp = ")
        str_list.append("trainset_int2utt = ")
        str_list.append("devset_feats_scp = ")
        str_list.append("devset_int2utt = ")
        str_list.append("testset_feats_scp = ")
        str_list.append("testset_int2utt = ")
        str_list.append("# int, min chunk size, default = 200")
        str_list.append("min_chunk_size = 200")
        str_list.append("# int, max chunk size, default = 400")
        str_list.append("max_chunk_size = 400")
        str_list.append("# int, min item size, default = 1")
        str_list.append("min_item_size = 1")
        str_list.append("# int, max item size, default = 1")
        str_list.append("max_item_size = 1")
        str_list.append("# int, repeat, only used in training procedure, default=50")
        str_list.append("repeat = 50")

        return str_list


# Model config
class XvectorParam(ParamBase):
    """
    Xvector parameter manager
    """
    def __init__(self, val_dict):
        super(XvectorParam, self).__init__(val_dict)
        self.format_params()

    def format_params(self):
        """
        Format the params
        """
        self.se_block_enable = True if self.se_block_enable.lower() == "true" else False
        self.se_block_r = int(self.se_block_r)

    @staticmethod
    def template():
        """
        Generate a string list of template
        """

        str_list = []
        str_list.append("[Xvector]")
        str_list.append("# bool, add senet or not, default = False")
        str_list.append("se_block_enable = 1")
        str_list.append("# int, senet reduction ratio, default = 16")
        str_list.append("se_block_r = 16")

        return str_list


class SelfAttXvectorParam(ParamBase):
    """
    SelfAttXvector parameter manager
    """
    def __init__(self, val_dict):
        super(SelfAttXvectorParam, self).__init__(val_dict)
        self.format_params()

    def format_params(self):
        """
        Format the params
        """
        self.head = int(self.head)
        self.alpha = float(self.alpha)
        self.use_std = True if self.use_std.lower() == "true" else False

    @staticmethod
    def template():
        """
        Generate a string list of template
        """

        str_list = []
        str_list.append("[SelfAttXvector]")
        str_list.append("# number of heads, default = 1")
        str_list.append("head = 1")
        str_list.append("# float, penalty, default = 0")
        str_list.append("alpha = 0")
        str_list.append("# bool, use std or not , default = True")
        str_list.append("use_std = True")

        return str_list


class ResnetParam(ParamBase):
    """
    Resnet parameter manager
    """
    def __init__(self, val_dict):
        super(ResnetParam, self).__init__(val_dict)
        self.format_params()

    def format_params(self):
        """
        Format the params
        """
        self.se_block_enable = True if self.se_block_enable.lower() == "true" else False
        self.se_block_r = int(self.se_block_r)

    @staticmethod
    def template():
        """
        Generate a string list of template
        """

        str_list = []
        str_list.append("[Resnet]")
        str_list.append("# bool, add senet or not, default = False")
        str_list.append("se_block_enable = 1")
        str_list.append("# int, senet reduction ratio, default = 16")
        str_list.append("se_block_r = 16")

        return str_list


# Loss config
class AMSoftmaxParam(ParamBase):
    """
    AMSoftmax parameter manager
    """
    def __init__(self, val_dict):
        super(AMSoftmaxParam, self).__init__(val_dict)
        self.format_params()

    def format_params(self):
        """
        Format the params
        """
        self.s = float(self.s)
        self.m = float(self.m)
        self.gamma = int(self.gamma)

    @staticmethod
    def template():
        """
        Generate a string list of template
        """

        str_list = []
        str_list.append("[AMSoftmax]")
        str_list.append("# Scale factor, default = 30")
        str_list.append("s = 30")
        str_list.append("# margin, default = 0.35")
        str_list.append("m = 0.35")
        str_list.append("# gamma for focal loss, int value, default = 5")
        str_list.append("gamma = 5")

        return str_list


class ASoftmaxParam(ParamBase):
    """
    ASoftmax parameter manager
    """
    def __init__(self, val_dict):
        super(ASoftmaxParam, self).__init__(val_dict)
        self.format_params()

    def format_params(self):
        """
        Format the params
        """
        self.phiflag = True if self.phiflag.lower() == "true" else False
        self.m = float(self.m)
        self.gamma = int(self.gamma)

    @staticmethod
    def template():
        """
        Generate a string list of template
        """

        str_list = []
        str_list.append("[ASoftmax]")
        str_list.append("# Bool, use customized phi function, default = False")
        str_list.append("phiflag = False")
        str_list.append("# margin, default = 4")
        str_list.append("m = 4")
        str_list.append("# gamma for focal loss, int value, default = 5")
        str_list.append("gamma = 5")

        return str_list


class FocalLossParam(ParamBase):
    """
    FocalLoss parameter manager
    """
    def __init__(self, val_dict):
        super(FocalLossParam, self).__init__(val_dict)
        self.format_params()

    def format_params(self):
        """
        Format the params
        """
        alpha = self.alpha.split(";")
        if len(alpha) == 1:
            self.alpha = float(alpha[0])
        else:
            self.alpha = [float(item) for item in alpha]
        self.gamma = int(self.gamma)

    @staticmethod
    def template():
        """
        Generate a string list of template
        """

        str_list = []
        str_list.append("[FocalLoss]")
        str_list.append("# float scalar or a list(seperate with ';', e.g 0.5;0.5), weight of each class,default = 1.0")
        str_list.append("alpha = 1.0")
        str_list.append("# int, gamma, default = 2")
        str_list.append("gamma = 2")

        return str_list


class GE2ELossParam(ParamBase):
    """
    GE2ELoss parameter manager
    """
    loss_method_list = ["softmax", "contrast"]

    def __init__(self, val_dict):
        super(GE2ELossParam, self).__init__(val_dict)
        self.format_params()

    def format_params(self):
        """
        Format the params
        """
        self.loss_method = self.loss_method.lower()

    def check(self):
        """
        check the parameters
        """
        if self.loss_method not in GE2ELossParam.loss_method_list:
            return False

        return True

    @staticmethod
    def template():
        """
        Generate a string list of template
        """

        str_list = []
        str_list.append("[GE2ELoss]")
        str_list.append("# loss method, value={%s}" % (", ".join(GE2ELossParam.loss_method_list)))
        str_list.append("loss_method = softmax")

        return str_list


class Config(object):
    """
    This class is used for get all configurations of configure file

     Attributes:
        file_path   :  存放配置的文件路径
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.param_cls = []

    def load_config(self):
        """
        load from configurations from conf_file
        """
        if not os.path.exists(self.file_path):
            logger.error("No such file: %s" % (self.file_path))
            return False

        config = configparser.ConfigParser()
        try:
            conf_res = config.read(self.file_path)
        except configparser.MissingSectionHeaderError as e:
            logger.error('Config-file error: %s' % e)
            return False
        except Exception as e:
            logger.error('Config-file error: %s' % e)
            return False

        if len(conf_res) == 0:
            return False

        names = self.__dict__
        for section in config.sections():
            cls_name = section + "Param"
            names[cls_name] = globals()[cls_name](config[section])
            self.param_cls.append(cls_name)

        return True

    def check(self):
        """
        check the parameters
        """
        names = self.__dict__
        success = True
        for cls_name in self.param_cls:
            if not names[cls_name].check():
                success = False
                break

        return success

    def dump_param(self):
        """
        Dump parameters to string
        """
        return "ToDo"

    @staticmethod
    def gen_templatefile(template_file):
        """
        Generate a template config file
        """
        with open(template_file, "w") as out_fh:
            out_fh.write("\n".join(CommonBaseParam.template()) + "\n\n")
            out_fh.write("\n".join(ModelCommonParam.template()) + "\n\n")
            out_fh.write("\n".join(TrainingParam.template()) + "\n\n")
            out_fh.write("\n".join(FinetuneParam.template()) + "\n\n")
            out_fh.write("\n".join(KaldiEgsDatasetParam.template()) + "\n\n")
            out_fh.write("\n".join(KaldiFeatsDatasetParam.template()) + "\n\n")
            out_fh.write("\n".join(XvectorParam.template()) + "\n\n")
            out_fh.write("\n".join(SelfAttXvectorParam.template()) + "\n\n")
            out_fh.write("\n".join(ResnetParam.template()) + "\n\n")
            out_fh.write("\n".join(AMSoftmaxParam.template()) + "\n\n")
            out_fh.write("\n".join(ASoftmaxParam.template()) + "\n\n")
            out_fh.write("\n".join(FocalLossParam.template()) + "\n\n")
            out_fh.write("\n".join(GE2ELossParam.template()) + "\n\n")


parser = argparse.ArgumentParser(description=__doc__, add_help=False)
parser.add_argument('-c', '--config_file', action='store', dest='CONF_PATH',
                    default=None, help='Set configuration file path')
parser.add_argument('-l', '--log_file', action='store', dest='LOG_PATH',
                    default=None, help='Set log file path')
parser.add_argument('-t', '--template_file', action='store', dest='TEMPLATE_PATH',
                    default=None, help='Set template configuration file path')

args, unkown_args = parser.parse_known_args()

if args.LOG_PATH:
    log_dir, log_name = os.path.split(args.LOG_PATH)
    logger = log.init_log(args.LOG_PATH, name=log_name, level=logging.INFO)
else:
    log_dir = None
    log_name = "default"
    logger = log.init_log(None, name=log_name, level=logging.INFO)

if args.TEMPLATE_PATH:
    Config.gen_templatefile(args.TEMPLATE_PATH)
    sys.exit(0)

conf = None
config_file = args.CONF_PATH
if args.CONF_PATH:
    conf = Config(config_file)
    if not conf.load_config() or not conf.check():
        logger.error("Something is wrong in the config file!")
        sys.exit(-1)
    conf.log_dir = log_dir
    conf.config_file = config_file
    conf.log_file = args.LOG_PATH
