# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
from paddle.fluid import core
from paddle.fluid import layers
from paddle.fluid.dygraph import base as imperative_base

from paddlespeech.s2t.utils.log import Log

__all__ = ["ClipGradByGlobalNormWithLog"]

logger = Log(__name__).getlog()


class ClipGradByGlobalNormWithLog(paddle.nn.ClipGradByGlobalNorm):
    def __init__(self, clip_norm):
        super().__init__(clip_norm)

    def __repr__(self):
        return f"{self.__class__.__name__}(global_clip_norm={self.clip_norm})"

    @imperative_base.no_grad
    def _dygraph_clip(self, params_grads):
        params_and_grads = []
        sum_square_list = []
        for i, (p, g) in enumerate(params_grads):
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                continue
            merge_grad = g
            if g.type == core.VarDesc.VarType.SELECTED_ROWS:
                merge_grad = layers.merge_selected_rows(g)
                merge_grad = layers.get_tensor_from_selected_rows(merge_grad)
            square = paddle.square(merge_grad)
            sum_square = layers.reduce_sum(square)
            sum_square_list.append(sum_square)

            # debug log, not dump all since slow down train process
            if i < 10:
                logger.debug(
                    f"Grad Before Clip: {p.name}: {float(sum_square.sqrt()) }")

        # all parameters have been filterd out
        if len(sum_square_list) == 0:
            return params_grads

        global_norm_var = layers.concat(sum_square_list)
        global_norm_var = layers.reduce_sum(global_norm_var)
        global_norm_var = layers.sqrt(global_norm_var)
        # debug log
        logger.debug(f"Grad Global Norm: {float(global_norm_var)}!!!!")

        max_global_norm = layers.fill_constant(
            shape=[1], dtype=global_norm_var.dtype, value=self.clip_norm)
        clip_var = layers.elementwise_div(
            x=max_global_norm,
            y=paddle.maximum(x=global_norm_var, y=max_global_norm))
        for i, (p, g) in enumerate(params_grads):
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                params_and_grads.append((p, g))
                continue
            new_grad = layers.elementwise_mul(x=g, y=clip_var)
            params_and_grads.append((p, new_grad))

            # debug log, not dump all since slow down train process
            if i < 10:
                logger.debug(
                    f"Grad After Clip: {p.name}: {float(new_grad.square().sum().sqrt())}"
                )

        return params_and_grads
