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
import collections.abc as collections_abc

import paddle

_i0A = [
    -4.41534164647933937950E-18, 3.33079451882223809783E-17,
    -2.43127984654795469359E-16, 1.71539128555513303061E-15,
    -1.16853328779934516808E-14, 7.67618549860493561688E-14,
    -4.85644678311192946090E-13, 2.95505266312963983461E-12,
    -1.72682629144155570723E-11, 9.67580903537323691224E-11,
    -5.18979560163526290666E-10, 2.65982372468238665035E-9,
    -1.30002500998624804212E-8, 6.04699502254191894932E-8,
    -2.67079385394061173391E-7, 1.11738753912010371815E-6,
    -4.41673835845875056359E-6, 1.64484480707288970893E-5,
    -5.75419501008210370398E-5, 1.88502885095841655729E-4,
    -5.76375574538582365885E-4, 1.63947561694133579842E-3,
    -4.32430999505057594430E-3, 1.05464603945949983183E-2,
    -2.37374148058994688156E-2, 4.93052842396707084878E-2,
    -9.49010970480476444210E-2, 1.71620901522208775349E-1,
    -3.04682672343198398683E-1, 6.76795274409476084995E-1
]

_i0B = [
    -7.23318048787475395456E-18, -4.83050448594418207126E-18,
    4.46562142029675999901E-17, 3.46122286769746109310E-17,
    -2.82762398051658348494E-16, -3.42548561967721913462E-16,
    1.77256013305652638360E-15, 3.81168066935262242075E-15,
    -9.55484669882830764870E-15, -4.15056934728722208663E-14,
    1.54008621752140982691E-14, 3.85277838274214270114E-13,
    7.18012445138366623367E-13, -1.79417853150680611778E-12,
    -1.32158118404477131188E-11, -3.14991652796324136454E-11,
    1.18891471078464383424E-11, 4.94060238822496958910E-10,
    3.39623202570838634515E-9, 2.26666899049817806459E-8,
    2.04891858946906374183E-7, 2.89137052083475648297E-6,
    6.88975834691682398426E-5, 3.36911647825569408990E-3,
    8.04490411014108831608E-1
]


def piecewise(x, condlist, funclist, *args, **kw):
    n2 = len(funclist)
    # n = len(condlist)
    n = 1
    if n == n2 - 1:  # compute the "otherwise" condition.
        condelse = ~paddle.any(condlist, axis=0, keepdim=True)
        condlist = paddle.concat([condlist, condelse], axis=0)
        n += 1
    elif n != n2:
        raise ValueError(
            "with {} condition(s), either {} or {} functions are expected"
            .format(n, n, n + 1))

    y = paddle.zeros(paddle.shape(x), x.dtype)
    for k in range(n):
        item = funclist[k]
        if not isinstance(item, collections_abc.Callable):
            y[condlist[k]] = item
        else:
            temp = condlist[k]
            if paddle.shape(x) == paddle.ones([1]):
                vals = x
                y = item(vals, *args, **kw)
            else:
                vals = x[temp]
                y[temp] = item(vals, *args, **kw)
    return y


def _chbevl(x, vals):
    b0 = vals[0]
    b1 = 0.0
    for i in range(1, len(vals)):
        b2 = b1
        b1 = b0
        b0 = x * b1 - b2 + vals[i]

    return 0.5 * (b0 - b2)


def _i0_1(x):
    out = paddle.exp(x) * _chbevl(x / 2.0 - 2, _i0A)
    return paddle.cast(out, dtype="float32")


def _i0_2(x):
    out = paddle.exp(x) * _chbevl(32.0 / x - 2.0, _i0B) / paddle.sqrt(x)
    return paddle.cast(out, dtype="float32")


def _i0_dispatcher(x):
    return (x, )


def i0(x):
    x = paddle.abs(x)
    condlist = x <= paddle.full([1], 8.0)
    condlist = condlist.unsqueeze(0)
    return piecewise(x, condlist, [_i0_1, _i0_2])


def _len_guards(M):
    """Handle small or incorrect window lengths"""
    if int(M) != M or M < 0:
        raise ValueError('Window length M must be a non-negative integer')
    return M <= 1


def _extend(M, sym):
    """Extend window by 1 sample if needed for DFT-even symmetry"""
    if not sym:
        return M + 1, True
    else:
        return M, False


def _truncate(w, needed):
    """Truncate window by 1 sample if needed for DFT-even symmetry"""
    if needed:
        return w[:-1]
    else:
        return w


def kaiser(M, beta, sym=True):
    if _len_guards(M):
        return paddle.ones(M)
    M, needs_trunc = _extend(M, sym)
    n = paddle.arange(0, M)
    alpha = (M - 1) / 2.0
    a = i0(beta * paddle.sqrt(1 - ((n - alpha) / alpha)**2.0))
    b = i0(paddle.full([1], beta))
    w = a / b
    return _truncate(w, needs_trunc)
