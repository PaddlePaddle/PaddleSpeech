from typing import Tuple

import paddle

from cosyvoice.utils.mask import make_pad_mask

############################## 相关utils函数，如下 ##############################

def _Tensor_max(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.maximum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.max(self, *args, **kwargs), paddle.argmax(self, *args, **kwargs)
        else:
            ret = paddle.max(self, *args, **kwargs)

    return ret

setattr(paddle.Tensor, "_max", _Tensor_max)
############################## 相关utils函数，如上 ##############################



class InterpolateRegulator(paddle.nn.Layer):
    def __init__(
        self,
        channels: int,
        sampling_ratios: Tuple,
        out_channels: int = None,
        groups: int = 1,
    ):
        super().__init__()
        self.sampling_ratios = sampling_ratios
        out_channels = out_channels or channels
        model = paddle.nn.LayerList(sublayers=[])
        if len(sampling_ratios) > 0:
            for _ in sampling_ratios:
                module = paddle.nn.Conv1d(channels, channels, 3, 1, 1)
                norm = paddle.nn.GroupNorm(num_groups=groups, num_channels=channels)
                act = paddle.nn.Mish()
                model.extend([module, norm, act])
        model.append(paddle.nn.Conv1d(channels, out_channels, 1, 1))
        self.model = paddle.nn.Sequential(*model)

    def forward(self, x, ylens=None):
        mask = (~make_pad_mask(ylens)).to(x).unsqueeze(-1)
        x = paddle.nn.functional.interpolate(
            x=x.transpose(1, 2).contiguous(), size=ylens._max(), mode="linear"
        )
        out = self.model(x).transpose(1, 2).contiguous()
        olens = ylens
        return out * mask, olens

    def inference(self, x1, x2, mel_len1, mel_len2, input_frame_rate=50):
        if x2.shape[1] > 40:
            x2_head = paddle.nn.functional.interpolate(
                x=x2[:, :20].transpose(1, 2).contiguous(),
                size=int(20 / input_frame_rate * 22050 / 256),
                mode="linear",
            )
            x2_mid = paddle.nn.functional.interpolate(
                x=x2[:, 20:-20].transpose(1, 2).contiguous(),
                size=mel_len2 - int(20 / input_frame_rate * 22050 / 256) * 2,
                mode="linear",
            )
            x2_tail = paddle.nn.functional.interpolate(
                x=x2[:, -20:].transpose(1, 2).contiguous(),
                size=int(20 / input_frame_rate * 22050 / 256),
                mode="linear",
            )
            x2 = paddle.cat([x2_head, x2_mid, x2_tail], dim=2)
        else:
            x2 = paddle.nn.functional.interpolate(
                x=x2.transpose(1, 2).contiguous(), size=mel_len2, mode="linear"
            )
        if x1.shape[1] != 0:
            x1 = paddle.nn.functional.interpolate(
                x=x1.transpose(1, 2).contiguous(), size=mel_len1, mode="linear"
            )
            x = paddle.cat([x1, x2], dim=2)
        else:
            x = x2
        out = self.model(x).transpose(1, 2).contiguous()
        return out, mel_len1 + mel_len2