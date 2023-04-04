# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import librosa.display
import matplotlib.pylab as plt

__all__ = [
    "plot_alignment",
    "plot_spectrogram",
    "plot_waveform",
    "plot_multihead_alignments",
    "plot_multilayer_multihead_alignments",
]


def plot_alignment(alignment, title=None):
    # alignment: [encoder_steps, decoder_steps)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment,
                   aspect='auto',
                   origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if title is not None:
        xlabel += '\n\n' + title
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    return fig


def plot_multihead_alignments(alignments, title=None):
    # alignments: [N, encoder_steps, decoder_steps)
    num_subplots = alignments.shape[0]

    fig, axes = plt.subplots(figsize=(6 * num_subplots, 4),
                             ncols=num_subplots,
                             sharey=True,
                             squeeze=True)
    for i, ax in enumerate(axes):
        im = ax.imshow(alignments[i],
                       aspect='auto',
                       origin='lower',
                       interpolation='none')
        fig.colorbar(im, ax=ax)
        xlabel = 'Decoder timestep'
        if title is not None:
            xlabel += '\n\n' + title
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel('Encoder timestep')
    plt.tight_layout()
    return fig


def plot_multilayer_multihead_alignments(alignments, title=None):
    # alignments: [num_layers, num_heads, encoder_steps, decoder_steps)
    num_layers, num_heads, *_ = alignments.shape

    fig, axes = plt.subplots(figsize=(6 * num_heads, 4 * num_layers),
                             nrows=num_layers,
                             ncols=num_heads,
                             sharex=True,
                             sharey=True,
                             squeeze=True)
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            im = ax.imshow(alignments[i, j],
                           aspect='auto',
                           origin='lower',
                           interpolation='none')
            fig.colorbar(im, ax=ax)
            xlabel = 'Decoder timestep'
            if title is not None:
                xlabel += '\n\n' + title
            if i == num_layers - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel('Encoder timestep')
    plt.tight_layout()
    return fig


def plot_spectrogram(spec):
    # spec: [C, T] librosa convention
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spec, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()
    return fig


def plot_waveform(wav, sr=22050):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = librosa.display.waveplot(wav, sr=22050)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig
