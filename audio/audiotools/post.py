# MIT License, Copyright (c) 2023-Present, Descript.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
# Modified from audiotools(https://github.com/descriptinc/audiotools/blob/master/audiotools/post.py)
import tempfile
import typing
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from audiotools import AudioSignal
import markdown2 as md
import matplotlib.pyplot as plt
import paddle
from IPython.display import HTML


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
    from audiotools import AudioSignal

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


def in_notebook():
    """Determines if code is running in a notebook.

    Returns
    -------
    bool
        Whether or not this is running in a notebook.
    """
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def disp(obj, **kwargs):
    """Displays an object, depending on if its in a notebook
    or not.

    Parameters
    ----------
    obj : typing.Any
        Any object to display.

    """

    IN_NOTEBOOK = in_notebook()

    if isinstance(obj, AudioSignal):
        audio_elem = obj.embed(display=False, return_html=True)
        if IN_NOTEBOOK:
            return HTML(audio_elem)
        else:
            print(audio_elem)
    if isinstance(obj, dict):
        table = audio_table(obj, **kwargs)
        if IN_NOTEBOOK:
            return HTML(md.markdown(table, extras=["tables"]))
        else:
            print(table)
    if isinstance(obj, plt.Figure):
        plt.show()
