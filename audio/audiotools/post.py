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
#
# This file contains code derived from [Descript]
# (https://github.com/descriptinc/audiotools),
# which is licensed under the MIT License:
#
# MIT License
#
# Copyright (c) 2023-Present, Descript
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import tempfile
import typing
import zipfile
from pathlib import Path

import markdown2 as md
import matplotlib.pyplot as plt
import paddle
from audiotools import AudioSignal
from IPython.display import HTML


def audio_table(
        audio_dict: dict,
        first_column: str=None,
        format_fn: typing.Callable=None,
        **kwargs, ):  # pragma: no cover
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


def in_notebook():  # pragma: no cover
    """Determines if code is running in a notebook.

    Returns
    -------
    bool
        Whether or not this is running in a notebook.
    """
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def disp(obj, **kwargs):  # pragma: no cover
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
