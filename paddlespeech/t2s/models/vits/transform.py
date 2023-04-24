# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""Flow-related transformation.

This code is based on https://github.com/bayesiains/nflows.

"""
import numpy as np
import paddle
from paddle.nn import functional as F

from paddlespeech.t2s.modules.nets_utils import paddle_gather

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def piecewise_rational_quadratic_transform(
        inputs,
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
        inverse=False,
        tails=None,
        tail_bound=1.0,
        min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE, ):
    if tails is None:
        spline_fn = rational_quadratic_spline
        spline_kwargs = {}
    else:
        spline_fn = unconstrained_rational_quadratic_spline
        spline_kwargs = {"tails": tails, "tail_bound": tail_bound}

    outputs, logabsdet = spline_fn(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
        **spline_kwargs)
    return outputs, logabsdet


def mask_preprocess(x, mask):
    B, C, T, bins = paddle.shape(x)
    new_x = paddle.zeros([mask.sum(), bins])
    for i in range(bins):
        new_x[:, i] = x[:, :, :, i][mask]
    return new_x


def unconstrained_rational_quadratic_spline(
        inputs,
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
        inverse=False,
        tails="linear",
        tail_bound=1.0,
        min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE, ):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = paddle.zeros(paddle.shape(inputs))
    logabsdet = paddle.zeros(paddle.shape(inputs))
    if tails == "linear":
        unnormalized_derivatives = F.pad(
            unnormalized_derivatives,
            pad=[0] * (len(unnormalized_derivatives.shape) - 1) * 2 + [1, 1])
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    unnormalized_widths = mask_preprocess(unnormalized_widths,
                                          inside_interval_mask)
    unnormalized_heights = mask_preprocess(unnormalized_heights,
                                           inside_interval_mask)
    unnormalized_derivatives = mask_preprocess(unnormalized_derivatives,
                                               inside_interval_mask)

    (outputs[inside_interval_mask],
     logabsdet[inside_interval_mask], ) = rational_quadratic_spline(
         inputs=inputs[inside_interval_mask],
         unnormalized_widths=unnormalized_widths,
         unnormalized_heights=unnormalized_heights,
         unnormalized_derivatives=unnormalized_derivatives,
         inverse=inverse,
         left=-tail_bound,
         right=tail_bound,
         bottom=-tail_bound,
         top=tail_bound,
         min_bin_width=min_bin_width,
         min_bin_height=min_bin_height,
         min_derivative=min_derivative, )

    return outputs, logabsdet


def rational_quadratic_spline(
        inputs,
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
        inverse=False,
        left=0.0,
        right=1.0,
        bottom=0.0,
        top=1.0,
        min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE, ):
    if paddle.min(inputs) < left or paddle.max(inputs) > right:
        raise ValueError("Input to a transform is not within its domain")

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, axis=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = paddle.cumsum(widths, axis=-1)
    cumwidths = F.pad(
        cumwidths,
        pad=[0] * (len(cumwidths.shape) - 1) * 2 + [1, 0],
        mode="constant",
        value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, axis=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = paddle.cumsum(heights, axis=-1)
    cumheights = F.pad(
        cumheights,
        pad=[0] * (len(cumheights.shape) - 1) * 2 + [1, 0],
        mode="constant",
        value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = _searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = _searchsorted(cumwidths, inputs)[..., None]
    input_cumwidths = paddle_gather(cumwidths, -1, bin_idx)[..., 0]
    input_bin_widths = paddle_gather(widths, -1, bin_idx)[..., 0]

    input_cumheights = paddle_gather(cumheights, -1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = paddle_gather(delta, -1, bin_idx)[..., 0]

    input_derivatives = paddle_gather(derivatives, -1, bin_idx)[..., 0]
    input_derivatives_plus_one = paddle_gather(derivatives[..., 1:], -1,
                                               bin_idx)[..., 0]

    input_heights = paddle_gather(heights, -1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta)
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - paddle.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta
             ) * theta_one_minus_theta)
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2) + 2 * input_delta *
            theta_one_minus_theta + input_derivatives * (1 - root).pow(2))
        logabsdet = paddle.log(derivative_numerator) - 2 * paddle.log(
            denominator)

        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2) +
                                     input_derivatives * theta_one_minus_theta)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta
             ) * theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2) + 2 * input_delta *
            theta_one_minus_theta + input_derivatives * (1 - theta).pow(2))
        logabsdet = paddle.log(derivative_numerator) - 2 * paddle.log(
            denominator)

        return outputs, logabsdet


def _searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return paddle.sum(inputs[..., None] >= bin_locations, axis=-1) - 1
