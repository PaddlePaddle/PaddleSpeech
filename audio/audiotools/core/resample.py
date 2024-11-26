import inspect
import math
from typing import Optional
from typing import Sequence

import paddle
import paddle.nn.functional as F


def simple_repr(obj, attrs: Optional[Sequence[str]]=None, overrides: dict={}):
    """
    Return a simple representation string for `obj`.
    If `attrs` is not None, it should be a list of attributes to include.
    """
    params = inspect.signature(obj.__class__).parameters
    attrs_repr = []
    if attrs is None:
        attrs = list(params.keys())
    for attr in attrs:
        display = False
        if attr in overrides:
            value = overrides[attr]
        elif hasattr(obj, attr):
            value = getattr(obj, attr)
        else:
            continue
        if attr in params:
            param = params[attr]
            if param.default is inspect._empty or value != param.default:  # type: ignore
                display = True
        else:
            display = True

        if display:
            attrs_repr.append(f"{attr}={value}")
    return f"{obj.__class__.__name__}({','.join(attrs_repr)})"


def sinc(x: paddle.Tensor):
    """
    Implementation of sinc, i.e. sin(x) / x

    __Warning__: the input is not multiplied by `pi`!
    """
    return paddle.where(
        x == 0,
        paddle.to_tensor(1.0, dtype=x.dtype, place=x.place),
        paddle.sin(x) / x, )


class ResampleFrac(paddle.nn.Layer):
    """
    Resampling from the sample rate `old_sr` to `new_sr`.
    """

    def __init__(self,
                 old_sr: int,
                 new_sr: int,
                 zeros: int=24,
                 rolloff: float=0.945):
        """
        Args:
            old_sr (int): sample rate of the input signal x.
            new_sr (int): sample rate of the output.
            zeros (int): number of zero crossing to keep in the sinc filter.
            rolloff (float): use a lowpass filter that is `rolloff * new_sr / 2`,
                to ensure sufficient margin due to the imperfection of the FIR filter used.
                Lowering this value will reduce anti-aliasing, but will reduce some of the
                highest frequencies.

        Shape:

            - Input: `[*, T]`
            - Output: `[*, T']` with `T' = int(new_sr * T / old_sr)`


        .. caution::
            After dividing `old_sr` and `new_sr` by their GCD, both should be small
            for this implementation to be fast.

        >>> import paddle
        >>> resample = ResampleFrac(4, 5)
        >>> x = paddle.randn([1000])
        >>> print(len(resample(x)))
        1250
        """
        super(ResampleFrac, self).__init__()
        if not isinstance(old_sr, int) or not isinstance(new_sr, int):
            raise ValueError("old_sr and new_sr should be integers")
        gcd = math.gcd(old_sr, new_sr)
        self.old_sr = old_sr // gcd
        self.new_sr = new_sr // gcd
        self.zeros = zeros
        self.rolloff = rolloff

        self._init_kernels()

    def _init_kernels(self):
        if self.old_sr == self.new_sr:
            return

        kernels = []
        sr = min(self.new_sr, self.old_sr)
        # rolloff will perform antialiasing filtering by removing the highest frequencies.
        # At first I thought I only needed this when downsampling, but when upsampling
        # you will get edge artifacts without this, the edge is equivalent to zero padding,
        # which will add high freq artifacts.
        sr *= self.rolloff

        # The key idea of the algorithm is that x(t) can be exactly reconstructed from x[i] (tensor)
        # using the sinc interpolation formula:
        #   x(t) = sum_i x[i] sinc(pi * old_sr * (i / old_sr - t))
        # We can then sample the function x(t) with a different sample rate:
        #    y[j] = x(j / new_sr)
        # or,
        #    y[j] = sum_i x[i] sinc(pi * old_sr * (i / old_sr - j / new_sr))

        # We see here that y[j] is the convolution of x[i] with a specific filter, for which
        # we take an FIR approximation, stopping when we see at least `zeros` zeros crossing.
        # But y[j+1] is going to have a different set of weights and so on, until y[j + new_sr].
        # Indeed:
        # y[j + new_sr] = sum_i x[i] sinc(pi * old_sr * ((i / old_sr - (j + new_sr) / new_sr))
        #               = sum_i x[i] sinc(pi * old_sr * ((i - old_sr) / old_sr - j / new_sr))
        #               = sum_i x[i + old_sr] sinc(pi * old_sr * (i / old_sr - j / new_sr))
        # so y[j+new_sr] uses the same filter as y[j], but on a shifted version of x by `old_sr`.
        # This will explain the F.conv1d after, with a stride of old_sr.
        self._width = math.ceil(self.zeros * self.old_sr / sr)
        # If old_sr is still big after GCD reduction, most filters will be very unbalanced, i.e.,
        # they will have a lot of almost zero values to the left or to the right...
        # There is probably a way to evaluate those filters more efficiently, but this is kept for
        # future work.
        idx = paddle.arange(
            -self._width, self._width + self.old_sr, dtype="float32")
        for i in range(self.new_sr):
            t = (-i / self.new_sr + idx / self.old_sr) * sr
            t = paddle.clip(t, -self.zeros, self.zeros)
            t *= math.pi
            window = paddle.cos(t / self.zeros / 2)**2
            kernel = sinc(t) * window
            # Renormalize kernel to ensure a constant signal is preserved.
            kernel = kernel / kernel.sum()
            kernels.append(kernel)

        _kernel = paddle.stack(kernels).reshape([self.new_sr, 1, -1])
        self.kernel = self.create_parameter(
            shape=_kernel.shape,
            dtype=_kernel.dtype, )
        self.kernel.set_value(_kernel)

    def forward(
            self,
            x: paddle.Tensor,
            output_length: Optional[int]=None,
            full: bool=False, ):
        """
        Resample x.
        Args:
            x (Tensor): signal to resample, time should be the last dimension
            output_length (None or int): This can be set to the desired output length
                (last dimension). Allowed values are between 0 and
                ceil(length * new_sr / old_sr). When None (default) is specified, the
                floored output length will be used. In order to select the largest possible
                size, use the `full` argument.
            full (bool): return the longest possible output from the input. This can be useful
                if you chain resampling operations, and want to give the `output_length` only
                for the last one, while passing `full=True` to all the other ones.
        """
        if self.old_sr == self.new_sr:
            return x
        shape = x.shape
        length = x.shape[-1]
        x = x.reshape([-1, length])
        x = F.pad(
            x.unsqueeze(1),
            [self._width, self._width + self.old_sr],
            mode="replicate",
            data_format="NCL", )
        ys = F.conv1d(x, self.kernel, stride=self.old_sr, data_format="NCL")
        y = ys.transpose([0, 2, 1]).reshape(list(shape[:-1]) + [-1])

        float_output_length = paddle.to_tensor(
            self.new_sr * length / self.old_sr, dtype="float32")
        max_output_length = paddle.ceil(float_output_length).astype("int64")
        default_output_length = paddle.floor(float_output_length).astype(
            "int64")

        if output_length is None:
            applied_output_length = (max_output_length
                                     if full else default_output_length)
        elif output_length < 0 or output_length > max_output_length:
            raise ValueError(
                f"output_length must be between 0 and {max_output_length.numpy()}"
            )
        else:
            applied_output_length = paddle.to_tensor(
                output_length, dtype="int64")
            if full:
                raise ValueError(
                    "You cannot pass both full=True and output_length")
        return y[..., :applied_output_length]

    def __repr__(self):
        return simple_repr(self)


def resample_frac(
        x: paddle.Tensor,
        old_sr: int,
        new_sr: int,
        zeros: int=24,
        rolloff: float=0.945,
        output_length: Optional[int]=None,
        full: bool=False, ):
    """
    Functional version of `ResampleFrac`, refer to its documentation for more information.

    ..warning::
        If you call repeatidly this functions with the same sample rates, then the
        resampling kernel will be recomputed everytime. For best performance, you should use
        and cache an instance of `ResampleFrac`.
    """
    return ResampleFrac(old_sr, new_sr, zeros, rolloff)(x, output_length, full)


if __name__ == "__main__":

    resample = ResampleFrac(4, 5)
    x = paddle.randn([1000])
    print(len(resample(x)))
