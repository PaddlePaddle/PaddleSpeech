#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from paddle.fluid import framework
from paddle.fluid import unique_name
from paddle.fluid.core import VarDesc
from paddle.fluid.initializer import MSRAInitializer

__all__ = ['KaimingUniform']


class KaimingUniform(MSRAInitializer):
    r"""Implements the Kaiming Uniform initializer

    This class implements the weight initialization from the paper
    `Delving Deep into Rectifiers: Surpassing Human-Level Performance on
    ImageNet Classification <https://arxiv.org/abs/1502.01852>`_
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun. This is a
    robust initialization method that particularly considers the rectifier
    nonlinearities.

    In case of Uniform distribution, the range is [-x, x], where

    .. math::

        x = \sqrt{\frac{1.0}{fan\_in}}

    In case of Normal distribution, the mean is 0 and the standard deviation
    is

    .. math::

        \sqrt{\\frac{2.0}{fan\_in}}

    Args:
        fan_in (float32|None): fan_in for Kaiming uniform Initializer. If None, it is\
        inferred from the variable. default is None.

    Note:
        It is recommended to set fan_in to None for most cases.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn

            linear = nn.Linear(2,
                               4,
                               weight_attr=nn.initializer.KaimingUniform())
            data = paddle.rand([30, 10, 2], dtype='float32')
            res = linear(data)

    """

    def __init__(self, fan_in=None):
        super(KaimingUniform, self).__init__(
            uniform=True, fan_in=fan_in, seed=0)

    def __call__(self, var, block=None):
        """Initialize the input tensor with MSRA initialization.

        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.

        Returns:
            The initialization op
        """
        block = self._check_block(block)

        assert isinstance(var, framework.Variable)
        assert isinstance(block, framework.Block)
        f_in, f_out = self._compute_fans(var)

        # If fan_in is passed, use it
        fan_in = f_in if self._fan_in is None else self._fan_in

        if self._seed == 0:
            self._seed = block.program.random_seed

        # to be compatible of fp16 initalizers
        if var.dtype == VarDesc.VarType.FP16 or (
                var.dtype == VarDesc.VarType.BF16 and not self._uniform):
            out_dtype = VarDesc.VarType.FP32
            out_var = block.create_var(
                name=unique_name.generate(
                    ".".join(['masra_init', var.name, 'tmp'])),
                shape=var.shape,
                dtype=out_dtype,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)
        else:
            out_dtype = var.dtype
            out_var = var

        if self._uniform:
            limit = np.sqrt(1.0 / float(fan_in))
            op = block.append_op(
                type="uniform_random",
                inputs={},
                outputs={"Out": out_var},
                attrs={
                    "shape": out_var.shape,
                    "dtype": int(out_dtype),
                    "min": -limit,
                    "max": limit,
                    "seed": self._seed
                },
                stop_gradient=True)

        else:
            std = np.sqrt(2.0 / float(fan_in))
            op = block.append_op(
                type="gaussian_random",
                outputs={"Out": out_var},
                attrs={
                    "shape": out_var.shape,
                    "dtype": int(out_dtype),
                    "mean": 0.0,
                    "std": std,
                    "seed": self._seed
                },
                stop_gradient=True)

        if var.dtype == VarDesc.VarType.FP16 or (
                var.dtype == VarDesc.VarType.BF16 and not self._uniform):
            block.append_op(
                type="cast",
                inputs={"X": out_var},
                outputs={"Out": var},
                attrs={"in_dtype": out_var.dtype,
                       "out_dtype": var.dtype})

        if not framework.in_dygraph_mode():
            var.op = op
        return op


class DefaultInitializerContext(object):
    """
        egs:
        with DefaultInitializerContext("kaiming_uniform"):
            code for setup_model
    """

    def __init__(self, init_type=None):
        self.init_type = init_type

    def __enter__(self):
        if self.init_type is None:
            return
        else:
            from paddlespeech.s2t.modules import align
            align.global_init_type = self.init_type
            return

    def __exit__(self, exc_type, exc_val, exc_tb):
        from paddlespeech.s2t.modules import align
        align.global_init_type = None
