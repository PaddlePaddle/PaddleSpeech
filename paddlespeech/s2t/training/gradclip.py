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
from paddle.fluid.clip import _squared_l2_norm
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
        sum_square_list_fp16 = []
        sum_square_list_fp32 = []
        for i, (p, g) in enumerate(params_grads):
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                continue
            merge_grad = g
            if g.type == core.VarDesc.VarType.SELECTED_ROWS:
                merge_grad = layers.merge_selected_rows(g)
                merge_grad = layers.get_tensor_from_selected_rows(merge_grad)

            sum_square = _squared_l2_norm(merge_grad)
            if sum_square.dtype == core.VarDesc.VarType.FP16:
                sum_square_list_fp16.append(sum_square)
            elif sum_square.dtype == core.VarDesc.VarType.FP32:
                sum_square_list_fp32.append(sum_square)
            else:
                sum_square_list.append(sum_square)

            # debug log, not dump all since slow down train process
            if i < 10:
                logger.debug(
                    f"Grad Before Clip: {p.name}: {float(sum_square.sqrt()) }")

        # all parameters have been filterd out
        if len(sum_square_list) + len(sum_square_list_fp16) + len(
                sum_square_list_fp32) == 0:
            return params_grads

        sum_dtype = 'float64' if len(sum_square_list) > 0 else "float32"
        global_norm_var = []
        if len(sum_square_list_fp16) > 0:
            global_norm_var_fp16 = paddle.add_n(sum_square_list_fp16)
            global_norm_var.append(global_norm_var_fp16.astype(sum_dtype))
        if len(sum_square_list_fp32) > 0:
            global_norm_var_fp32 = paddle.add_n(sum_square_list_fp32)
            if sum_dtype == 'float32':
                global_norm_var.append(global_norm_var_fp32)
            else:
                global_norm_var.append(global_norm_var_fp32.astype(sum_dtype))
        if len(sum_square_list) > 0:
            global_norm_var_fp64 = paddle.add_n(sum_square_list)
            global_norm_var.append(global_norm_var_fp64)
        global_norm_var = paddle.add_n(global_norm_var)
        global_norm_var = layers.sqrt(global_norm_var)
        logger.debug(f"Grad Global Norm: {float(global_norm_var)}!!!!")

        max_global_norm = layers.fill_constant(
            shape=[1], dtype=global_norm_var.dtype, value=self.clip_norm)

        # only when global_norm_var > max_global_norm, grad need clip
        need_clip = False
        if global_norm_var > max_global_norm:
            need_clip = True

        if need_clip:
            clip_var = layers.elementwise_div(
                x=max_global_norm, y=global_norm_var)
        for i, (p, g) in enumerate(params_grads):
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                params_and_grads.append((p, g))
                continue
            # TODO(wangxi): use inplace elementwise_mul
            if need_clip:
                clip_input = (clip_var.astype('float16')
                              if g.dtype == core.VarDesc.VarType.FP16 else
                              clip_var)
                new_grad = layers.elementwise_mul(x=g, y=clip_input)
                params_and_grads.append((p, new_grad))
                # debug log, not dump all since slow down train process
                if i < 10:
                    logger.debug(
                        f"Grad After Clip: {p.name}: {float(new_grad.square().sum().sqrt())}"
                    )
            else:
                params_and_grads.append((p, g))

        return params_and_grads
