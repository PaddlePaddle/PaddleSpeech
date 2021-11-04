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
# Modified from chainer(https://github.com/chainer/chainer)
import copy
import os

import numpy as np

from . import extension


class PlotAttentionReport(extension.Extension):
    """Plot attention reporter.

    Args:
        att_vis_fn (espnet.nets.*_backend.e2e_asr.E2E.calculate_all_attentions):
            Function of attention visualization.
        data (list[tuple(str, dict[str, list[Any]])]): List json utt key items.
        outdir (str): Directory to save figures.
        converter (espnet.asr.*_backend.asr.CustomConverter):
            Function to convert data.
        device (int | torch.device): Device.
        reverse (bool): If True, input and output length are reversed.
        ikey (str): Key to access input
            (for ASR/ST ikey="input", for MT ikey="output".)
        iaxis (int): Dimension to access input
            (for ASR/ST iaxis=0, for MT iaxis=1.)
        okey (str): Key to access output
            (for ASR/ST okey="input", MT okay="output".)
        oaxis (int): Dimension to access output
            (for ASR/ST oaxis=0, for MT oaxis=0.)
        subsampling_factor (int): subsampling factor in encoder

    """

    def __init__(
            self,
            att_vis_fn,
            data,
            outdir,
            converter,
            transform,
            device,
            reverse=False,
            ikey="input",
            iaxis=0,
            okey="output",
            oaxis=0,
            subsampling_factor=1, ):
        self.att_vis_fn = att_vis_fn
        self.data = copy.deepcopy(data)
        self.data_dict = {k: v for k, v in copy.deepcopy(data)}
        # key is utterance ID
        self.outdir = outdir
        self.converter = converter
        self.transform = transform
        self.device = device
        self.reverse = reverse
        self.ikey = ikey
        self.iaxis = iaxis
        self.okey = okey
        self.oaxis = oaxis
        self.factor = subsampling_factor
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def __call__(self, trainer):
        """Plot and save image file of att_ws matrix."""
        att_ws, uttid_list = self.get_attention_weights()
        if isinstance(att_ws, list):  # multi-encoder case
            num_encs = len(att_ws) - 1
            # atts
            for i in range(num_encs):
                for idx, att_w in enumerate(att_ws[i]):
                    filename = "%s/%s.ep.{.updater.epoch}.att%d.png" % (
                        self.outdir, uttid_list[idx], i + 1, )
                    att_w = self.trim_attention_weight(uttid_list[idx], att_w)
                    np_filename = "%s/%s.ep.{.updater.epoch}.att%d.npy" % (
                        self.outdir, uttid_list[idx], i + 1, )
                    np.save(np_filename.format(trainer), att_w)
                    self._plot_and_save_attention(att_w,
                                                  filename.format(trainer))
            # han
            for idx, att_w in enumerate(att_ws[num_encs]):
                filename = "%s/%s.ep.{.updater.epoch}.han.png" % (
                    self.outdir, uttid_list[idx], )
                att_w = self.trim_attention_weight(uttid_list[idx], att_w)
                np_filename = "%s/%s.ep.{.updater.epoch}.han.npy" % (
                    self.outdir, uttid_list[idx], )
                np.save(np_filename.format(trainer), att_w)
                self._plot_and_save_attention(
                    att_w, filename.format(trainer), han_mode=True)
        else:
            for idx, att_w in enumerate(att_ws):
                filename = "%s/%s.ep.{.updater.epoch}.png" % (self.outdir,
                                                              uttid_list[idx], )
                att_w = self.trim_attention_weight(uttid_list[idx], att_w)
                np_filename = "%s/%s.ep.{.updater.epoch}.npy" % (
                    self.outdir, uttid_list[idx], )
                np.save(np_filename.format(trainer), att_w)
                self._plot_and_save_attention(att_w, filename.format(trainer))

    def log_attentions(self, logger, step):
        """Add image files of att_ws matrix to the tensorboard."""
        att_ws, uttid_list = self.get_attention_weights()
        if isinstance(att_ws, list):  # multi-encoder case
            num_encs = len(att_ws) - 1
            # atts
            for i in range(num_encs):
                for idx, att_w in enumerate(att_ws[i]):
                    att_w = self.trim_attention_weight(uttid_list[idx], att_w)
                    plot = self.draw_attention_plot(att_w)
                    logger.add_figure(
                        "%s_att%d" % (uttid_list[idx], i + 1),
                        plot.gcf(),
                        step, )
            # han
            for idx, att_w in enumerate(att_ws[num_encs]):
                att_w = self.trim_attention_weight(uttid_list[idx], att_w)
                plot = self.draw_han_plot(att_w)
                logger.add_figure(
                    "%s_han" % (uttid_list[idx]),
                    plot.gcf(),
                    step, )
        else:
            for idx, att_w in enumerate(att_ws):
                att_w = self.trim_attention_weight(uttid_list[idx], att_w)
                plot = self.draw_attention_plot(att_w)
                logger.add_figure("%s" % (uttid_list[idx]), plot.gcf(), step)

    def get_attention_weights(self):
        """Return attention weights.

        Returns:
            numpy.ndarray: attention weights. float. Its shape would be
                differ from backend.
                * pytorch-> 1) multi-head case => (B, H, Lmax, Tmax), 2)
                    other case => (B, Lmax, Tmax).
                * chainer-> (B, Lmax, Tmax)

        """
        return_batch, uttid_list = self.transform(self.data, return_uttid=True)
        batch = self.converter([return_batch], self.device)
        if isinstance(batch, tuple):
            att_ws = self.att_vis_fn(*batch)
        else:
            att_ws = self.att_vis_fn(**batch)
        return att_ws, uttid_list

    def trim_attention_weight(self, uttid, att_w):
        """Transform attention matrix with regard to self.reverse."""
        if self.reverse:
            enc_key, enc_axis = self.okey, self.oaxis
            dec_key, dec_axis = self.ikey, self.iaxis
        else:
            enc_key, enc_axis = self.ikey, self.iaxis
            dec_key, dec_axis = self.okey, self.oaxis
        dec_len = int(self.data_dict[uttid][dec_key][dec_axis]["shape"][0])
        enc_len = int(self.data_dict[uttid][enc_key][enc_axis]["shape"][0])
        if self.factor > 1:
            enc_len //= self.factor
        if len(att_w.shape) == 3:
            att_w = att_w[:, :dec_len, :enc_len]
        else:
            att_w = att_w[:dec_len, :enc_len]
        return att_w

    def draw_attention_plot(self, att_w):
        """Plot the att_w matrix.

        Returns:
            matplotlib.pyplot: pyplot object with attention matrix image.

        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.clf()
        att_w = att_w.astype(np.float32)
        if len(att_w.shape) == 3:
            for h, aw in enumerate(att_w, 1):
                plt.subplot(1, len(att_w), h)
                plt.imshow(aw, aspect="auto")
                plt.xlabel("Encoder Index")
                plt.ylabel("Decoder Index")
        else:
            plt.imshow(att_w, aspect="auto")
            plt.xlabel("Encoder Index")
            plt.ylabel("Decoder Index")
        plt.tight_layout()
        return plt

    def draw_han_plot(self, att_w):
        """Plot the att_w matrix for hierarchical attention.

        Returns:
            matplotlib.pyplot: pyplot object with attention matrix image.

        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.clf()
        if len(att_w.shape) == 3:
            for h, aw in enumerate(att_w, 1):
                legends = []
                plt.subplot(1, len(att_w), h)
                for i in range(aw.shape[1]):
                    plt.plot(aw[:, i])
                    legends.append("Att{}".format(i))
                plt.ylim([0, 1.0])
                plt.xlim([0, aw.shape[0]])
                plt.grid(True)
                plt.ylabel("Attention Weight")
                plt.xlabel("Decoder Index")
                plt.legend(legends)
        else:
            legends = []
            for i in range(att_w.shape[1]):
                plt.plot(att_w[:, i])
                legends.append("Att{}".format(i))
            plt.ylim([0, 1.0])
            plt.xlim([0, att_w.shape[0]])
            plt.grid(True)
            plt.ylabel("Attention Weight")
            plt.xlabel("Decoder Index")
            plt.legend(legends)
        plt.tight_layout()
        return plt

    def _plot_and_save_attention(self, att_w, filename, han_mode=False):
        if han_mode:
            plt = self.draw_han_plot(att_w)
        else:
            plt = self.draw_attention_plot(att_w)
        plt.savefig(filename)
        plt.close()


class PlotCTCReport(extension.Extension):
    """Plot CTC reporter.

    Args:
        ctc_vis_fn (espnet.nets.*_backend.e2e_asr.E2E.calculate_all_ctc_probs):
            Function of CTC visualization.
        data (list[tuple(str, dict[str, list[Any]])]): List json utt key items.
        outdir (str): Directory to save figures.
        converter (espnet.asr.*_backend.asr.CustomConverter):
            Function to convert data.
        device (int | torch.device): Device.
        reverse (bool): If True, input and output length are reversed.
        ikey (str): Key to access input
            (for ASR/ST ikey="input", for MT ikey="output".)
        iaxis (int): Dimension to access input
            (for ASR/ST iaxis=0, for MT iaxis=1.)
        okey (str): Key to access output
            (for ASR/ST okey="input", MT okay="output".)
        oaxis (int): Dimension to access output
            (for ASR/ST oaxis=0, for MT oaxis=0.)
        subsampling_factor (int): subsampling factor in encoder

    """

    def __init__(
            self,
            ctc_vis_fn,
            data,
            outdir,
            converter,
            transform,
            device,
            reverse=False,
            ikey="input",
            iaxis=0,
            okey="output",
            oaxis=0,
            subsampling_factor=1, ):
        self.ctc_vis_fn = ctc_vis_fn
        self.data = copy.deepcopy(data)
        self.data_dict = {k: v for k, v in copy.deepcopy(data)}
        # key is utterance ID
        self.outdir = outdir
        self.converter = converter
        self.transform = transform
        self.device = device
        self.reverse = reverse
        self.ikey = ikey
        self.iaxis = iaxis
        self.okey = okey
        self.oaxis = oaxis
        self.factor = subsampling_factor
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def __call__(self, trainer):
        """Plot and save image file of ctc prob."""
        ctc_probs, uttid_list = self.get_ctc_probs()
        if isinstance(ctc_probs, list):  # multi-encoder case
            num_encs = len(ctc_probs) - 1
            for i in range(num_encs):
                for idx, ctc_prob in enumerate(ctc_probs[i]):
                    filename = "%s/%s.ep.{.updater.epoch}.ctc%d.png" % (
                        self.outdir, uttid_list[idx], i + 1, )
                    ctc_prob = self.trim_ctc_prob(uttid_list[idx], ctc_prob)
                    np_filename = "%s/%s.ep.{.updater.epoch}.ctc%d.npy" % (
                        self.outdir, uttid_list[idx], i + 1, )
                    np.save(np_filename.format(trainer), ctc_prob)
                    self._plot_and_save_ctc(ctc_prob, filename.format(trainer))
        else:
            for idx, ctc_prob in enumerate(ctc_probs):
                filename = "%s/%s.ep.{.updater.epoch}.png" % (self.outdir,
                                                              uttid_list[idx], )
                ctc_prob = self.trim_ctc_prob(uttid_list[idx], ctc_prob)
                np_filename = "%s/%s.ep.{.updater.epoch}.npy" % (
                    self.outdir, uttid_list[idx], )
                np.save(np_filename.format(trainer), ctc_prob)
                self._plot_and_save_ctc(ctc_prob, filename.format(trainer))

    def log_ctc_probs(self, logger, step):
        """Add image files of ctc probs to the tensorboard."""
        ctc_probs, uttid_list = self.get_ctc_probs()
        if isinstance(ctc_probs, list):  # multi-encoder case
            num_encs = len(ctc_probs) - 1
            for i in range(num_encs):
                for idx, ctc_prob in enumerate(ctc_probs[i]):
                    ctc_prob = self.trim_ctc_prob(uttid_list[idx], ctc_prob)
                    plot = self.draw_ctc_plot(ctc_prob)
                    logger.add_figure(
                        "%s_ctc%d" % (uttid_list[idx], i + 1),
                        plot.gcf(),
                        step, )
        else:
            for idx, ctc_prob in enumerate(ctc_probs):
                ctc_prob = self.trim_ctc_prob(uttid_list[idx], ctc_prob)
                plot = self.draw_ctc_plot(ctc_prob)
                logger.add_figure("%s" % (uttid_list[idx]), plot.gcf(), step)

    def get_ctc_probs(self):
        """Return CTC probs.

        Returns:
            numpy.ndarray: CTC probs. float. Its shape would be
                differ from backend. (B, Tmax, vocab).

        """
        return_batch, uttid_list = self.transform(self.data, return_uttid=True)
        batch = self.converter([return_batch], self.device)
        if isinstance(batch, tuple):
            probs = self.ctc_vis_fn(*batch)
        else:
            probs = self.ctc_vis_fn(**batch)
        return probs, uttid_list

    def trim_ctc_prob(self, uttid, prob):
        """Trim CTC posteriors accoding to input lengths."""
        enc_len = int(self.data_dict[uttid][self.ikey][self.iaxis]["shape"][0])
        if self.factor > 1:
            enc_len //= self.factor
        prob = prob[:enc_len]
        return prob

    def draw_ctc_plot(self, ctc_prob):
        """Plot the ctc_prob matrix.

        Returns:
            matplotlib.pyplot: pyplot object with CTC prob matrix image.

        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ctc_prob = ctc_prob.astype(np.float32)

        plt.clf()
        topk_ids = np.argsort(ctc_prob, axis=1)
        n_frames, vocab = ctc_prob.shape
        times_probs = np.arange(n_frames)

        plt.figure(figsize=(20, 8))

        # NOTE: index 0 is reserved for blank
        for idx in set(topk_ids.reshape(-1).tolist()):
            if idx == 0:
                plt.plot(
                    times_probs,
                    ctc_prob[:, 0],
                    ":",
                    label="<blank>",
                    color="grey")
            else:
                plt.plot(times_probs, ctc_prob[:, idx])
        plt.xlabel(u"Input [frame]", fontsize=12)
        plt.ylabel("Posteriors", fontsize=12)
        plt.xticks(list(range(0, int(n_frames) + 1, 10)))
        plt.yticks(list(range(0, 2, 1)))
        plt.tight_layout()
        return plt

    def _plot_and_save_ctc(self, ctc_prob, filename):
        plt = self.draw_ctc_plot(ctc_prob)
        plt.savefig(filename)
        plt.close()
