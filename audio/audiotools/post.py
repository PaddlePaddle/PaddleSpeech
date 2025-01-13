# MIT License, Copyright (c) 2023-Present, Descript.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Modified from audiotools(https://github.com/descriptinc/audiotools/blob/master/audiotools/post.py)
import typing

import paddle

from audio.audiotools.core import AudioSignal


def audio_table(
        audio_dict: dict,
        first_column: str=None,
        format_fn: typing.Callable=None,
        **kwargs, ):
    """Embeds an audio table into HTML, or as the output cell
    in a notebook.

    Parameters
    ----------
    audio_dict : dict
        Dictionary of data to embed.
    first_column : str, optional
        The label for the first column of the table, by default None
    format_fn : typing.Callable, optional
        How to format the data, by default None

    Returns
    -------
    str
        Table as a string

    Examples
    --------

    >>> audio_dict = {}
    >>> for i in range(signal_batch.batch_size):
    >>>     audio_dict[i] = {
    >>>         "input": signal_batch[i],
    >>>         "output": output_batch[i]
    >>>     }
    >>> audiotools.post.audio_zip(audio_dict)

    """

    output = []
    columns = None

    def _default_format_fn(label, x, **kwargs):
        if paddle.is_tensor(x):
            x = x.tolist()

        if x is None:
            return "."
        elif isinstance(x, AudioSignal):
            return x.embed(display=False, return_html=True, **kwargs)
        else:
            return str(x)

    if format_fn is None:
        format_fn = _default_format_fn

    if first_column is None:
        first_column = "."

    for k, v in audio_dict.items():
        if not isinstance(v, dict):
            v = {"Audio": v}

        v_keys = list(v.keys())
        if columns is None:
            columns = [first_column] + v_keys
            output.append(" | ".join(columns))

            layout = "|---" + len(v_keys) * "|:-:"
            output.append(layout)

        formatted_audio = []
        for col in columns[1:]:
            formatted_audio.append(format_fn(col, v[col], **kwargs))

        row = f"| {k} | "
        row += " | ".join(formatted_audio)
        output.append(row)

    output = "\n" + "\n".join(output)
    return output
