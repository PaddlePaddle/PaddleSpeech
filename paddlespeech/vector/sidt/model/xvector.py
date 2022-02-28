#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright     2020    Zeng Xingui(zengxingui@baidu.com)
#
########################################################################

"""
Xvector modle which defined in the paper
"X-VECTORS: ROBUST DNN EMBEDDINGS FOR SPEAKER RECOGNITION"
"""

import argparse
import paddle.nn as nn

from sidt.layer.layers import StatisticsPooling


class Xvector(nn.Layer):
    """
    Xvector modle which defined in the paper
    "X-VECTORS: ROBUST DNN EMBEDDINGS FOR SPEAKER RECOGNITION"
    """

    def __init__(self,
                 in_channels=23,
                 out_neurons=1,
                 activation=nn.LeakyReLU,
                 final_linear=nn.Linear,
                 tdnn_blocks=5,
                 tdnn_channels=[512, 512, 512, 512, 1500],
                 tdnn_kernel_sizes=[5, 3, 3, 1, 1],
                 tdnn_dilations=[1, 2, 3, 1, 1],
                 lin_neurons=512,
                 mode="train"):
        super(Xvector, self).__init__()

        assert in_channels > 0 and out_neurons > 0
        self.blocks = nn.LayerList()

        for block_index in range(tdnn_blocks):
            out_channels = tdnn_channels[block_index]
            self.blocks.extend(
                [
                    nn.Conv1D(
                        in_channels,
                        out_channels,
                        kernel_size=tdnn_kernel_sizes[block_index],
                        dilation=tdnn_dilations[block_index],
                    ),
                    activation(),
                    nn.BatchNorm1D(out_channels),
                ]
            )
            in_channels = tdnn_channels[block_index]

        self.blocks.append(StatisticsPooling())
        self.blocks.append(nn.Linear(out_channels * 2, lin_neurons))
        if mode != "xvector-fc1":
            self.blocks.append(activation())
            self.blocks.append(nn.BatchNorm1D(lin_neurons))
            self.blocks.append(nn.Linear(lin_neurons, lin_neurons))
            if mode != "xvector-fc2":
                self.blocks.append(activation())
                self.blocks.append(nn.BatchNorm1D(lin_neurons))
                if mode != "xvector-bn7":
                    self.blocks.append(final_linear(lin_neurons, out_neurons))

    #@paddle.jit.to_static
    def forward(self, x, lens=None):
        """
        Forward inference for the model

        Args:
            x: input tensor

        Returns:
            x: output tensor
        """
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lens)
            except TypeError:
                x = layer(x)

        return x

    @staticmethod
    def add_specific_args(parent_parser):
        """
        Static class method for xvector parameters configuration

        Args:
            parent_parser: instance of argparse.ArgumentParser

        Returns:
            parsers: instance of argparse.ArgumentParser
        """
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--n_in", action="store", type=int,
                            help="n_in(int): input feature dim")
        parser.add_argument("--n_out", action="store", type=int,
                            help="n_out(int): output feature dim")

        return parser
