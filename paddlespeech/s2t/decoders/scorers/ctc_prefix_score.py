#!/usr/bin/env python3
# Copyright 2018 Mitsubishi Electric Research Labs (Takaaki Hori)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import paddle
import six


class CTCPrefixScorePD():
    """Batch processing of CTCPrefixScore

    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    but extended to efficiently compute the label probabilities for multiple
    hypotheses simultaneously
    See also Seki et al. "Vectorized Beam Search for CTC-Attention-Based
    Speech Recognition," In INTERSPEECH (pp. 3825-3829), 2019.
    """

    def __init__(self, x, xlens, blank, eos, margin=0):
        """Construct CTC prefix scorer

        `margin` is M in eq.(22,23)

        :param paddle.Tensor x: input label posterior sequences (B, T, O)
        :param paddle.Tensor xlens: input lengths (B,)
        :param int blank: blank label id
        :param int eos: end-of-sequence id
        :param int margin: margin parameter for windowing (0 means no windowing)
        """
        # In the comment lines,
        # we assume T: input_length, B: batch size, W: beam width, O: output dim.
        self.logzero = -10000000000.0
        self.blank = blank
        self.eos = eos
        self.batch = paddle.shape(x)[0]
        self.input_length = paddle.shape(x)[1]
        self.odim = paddle.shape(x)[2]
        self.dtype = x.dtype

        # Pad the rest of posteriors in the batch
        # TODO(takaaki-hori): need a better way without for-loops
        for i, l in enumerate(xlens):
            if l < self.input_length:
                x[i, l:, :] = self.logzero
                x[i, l:, blank] = 0
        # Reshape input x
        xn = x.transpose([1, 0, 2])  # (B, T, O) -> (T, B, O)
        xb = xn[:, :, self.blank].unsqueeze(2).expand(-1, -1,
                                                      self.odim)  # (T,B,O)
        self.x = paddle.stack([xn, xb])  # (2, T, B, O)
        self.end_frames = paddle.to_tensor(xlens) - 1  # (B,)

        # Setup CTC windowing
        self.margin = margin
        if margin > 0:
            self.frame_ids = paddle.arange(self.input_length, dtype=self.dtype)
        # Base indices for index conversion
        # B idx, hyp idx. shape (B*W, 1)
        self.idx_bh = None
        # B idx. shape (B,)
        self.idx_b = paddle.arange(self.batch)
        # B idx, O idx. shape (B, 1)
        self.idx_bo = (self.idx_b * self.odim).unsqueeze(1)

    def __call__(self, y, state, scoring_ids=None, att_w=None):
        """Compute CTC prefix scores for next labels

        :param list y: prefix label sequences
        :param tuple state: previous CTC state
        :param paddle.Tensor scoring_ids: selected next ids to score (BW, O'), O' <= O
        :param paddle.Tensor att_w: attention weights to decide CTC window
        :return new_state, ctc_local_scores (BW, O)
        """
        output_length = len(y[0]) - 1  # ignore sos
        last_ids = [yi[-1] for yi in y]  # last output label ids
        n_bh = len(last_ids)  # batch * hyps
        n_hyps = n_bh // self.batch  # assuming each utterance has the same # of hyps
        self.scoring_num = paddle.shape(scoring_ids)[
            -1] if scoring_ids is not None else 0
        # prepare state info
        if state is None:
            r_prev = paddle.full(
                (self.input_length, 2, self.batch, n_hyps),
                self.logzero,
                dtype=self.dtype, )  # (T, 2, B, W)
            r_prev[:, 1] = paddle.cumsum(self.x[0, :, :, self.blank],
                                         0).unsqueeze(2)
            r_prev = r_prev.reshape([-1, 2, n_bh])  # (T, 2, BW)
            s_prev = 0.0  # score
            f_min_prev = 0  # eq. 22-23
            f_max_prev = 1  # eq. 22-23
        else:
            r_prev, s_prev, f_min_prev, f_max_prev = state

        # select input dimensions for scoring
        if self.scoring_num > 0:
            # (BW, O)
            scoring_idmap = paddle.full(
                (n_bh, self.odim), -1, dtype=paddle.long)
            snum = self.scoring_num
            if self.idx_bh is None or n_bh > len(self.idx_bh):
                self.idx_bh = paddle.arange(n_bh).reshape([-1, 1])  # (BW, 1)
            scoring_idmap[self.idx_bh[:n_bh], scoring_ids] = paddle.arange(snum)
            scoring_idx = (
                scoring_ids + self.idx_bo.repeat(1, n_hyps).reshape(
                    [-1, 1])  # (BW,1)
            ).reshape([-1])  # (BWO)
            # x_ shape (2, T, B*W, O)
            x_ = paddle.index_select(
                self.x.reshape([2, -1, self.batch * self.odim]), scoring_idx,
                2).reshape([2, -1, n_bh, snum])
        else:
            scoring_ids = None
            scoring_idmap = None
            snum = self.odim
            # x_ shape (2, T, B*W, O)
            x_ = self.x.unsqueeze(3).repeat(1, 1, 1, n_hyps, 1).reshape(
                [2, -1, n_bh, snum])

        # new CTC forward probs are prepared as a (T x 2 x BW x S) tensor
        # that corresponds to r_t^n(h) and r_t^b(h) in a batch.
        r = paddle.full(
            (self.input_length, 2, n_bh, snum),
            self.logzero,
            dtype=self.dtype, )
        if output_length == 0:
            r[0, 0] = x_[0, 0]

        r_sum = paddle.logsumexp(r_prev, 1)  #(T,BW)
        log_phi = r_sum.unsqueeze(2).repeat(1, 1, snum)  # (T, BW, O)
        if scoring_ids is not None:
            for idx in range(n_bh):
                pos = scoring_idmap[idx, last_ids[idx]]
                if pos >= 0:
                    log_phi[:, idx, pos] = r_prev[:, 1, idx]
        else:
            for idx in range(n_bh):
                log_phi[:, idx, last_ids[idx]] = r_prev[:, 1, idx]

        # decide start and end frames based on attention weights
        if att_w is not None and self.margin > 0:
            f_arg = paddle.matmul(att_w, self.frame_ids)
            f_min = max(int(f_arg.min().cpu()), f_min_prev)
            f_max = max(int(f_arg.max().cpu()), f_max_prev)
            start = min(f_max_prev, max(f_min - self.margin, output_length, 1))
            end = min(f_max + self.margin, self.input_length)
        else:
            f_min = f_max = 0
            # if one frame one out, the output_length is the eating frame num now.
            start = max(output_length, 1)
            end = self.input_length

        # compute forward probabilities log(r_t^n(h)) and log(r_t^b(h))
        for t in range(start, end):
            rp = r[t - 1]  # (2 x BW x O')
            rr = paddle.stack([rp[0], log_phi[t - 1], rp[0], rp[1]]).reshape(
                [2, 2, n_bh, snum])  # (2,2,BW,O')
            r[t] = paddle.logsumexp(rr, 1) + x_[:, t]

        # compute log prefix probabilities log(psi)
        log_phi_x = paddle.concat(
            (log_phi[0].unsqueeze(0), log_phi[:-1]), axis=0) + x_[0]
        if scoring_ids is not None:
            log_psi = paddle.full(
                (n_bh, self.odim), self.logzero, dtype=self.dtype)
            log_psi_ = paddle.logsumexp(
                paddle.concat(
                    (log_phi_x[start:end], r[start - 1, 0].unsqueeze(0)),
                    axis=0),
                axis=0, )
            for si in range(n_bh):
                log_psi[si, scoring_ids[si]] = log_psi_[si]
        else:
            log_psi = paddle.logsumexp(
                paddle.concat(
                    (log_phi_x[start:end], r[start - 1, 0].unsqueeze(0)),
                    axis=0),
                axis=0, )

        for si in range(n_bh):
            log_psi[si, self.eos] = r_sum[self.end_frames[si // n_hyps], si]

        # exclude blank probs
        log_psi[:, self.blank] = self.logzero

        return (log_psi - s_prev), (r, log_psi, f_min, f_max, scoring_idmap)

    def index_select_state(self, state, best_ids):
        """Select CTC states according to best ids

        :param state    : CTC state
        :param best_ids : index numbers selected by beam pruning (B, W)
        :return selected_state
        """
        r, s, f_min, f_max, scoring_idmap = state
        # convert ids to BHO space
        n_bh = len(s)
        n_hyps = n_bh // self.batch
        vidx = (best_ids +
                (self.idx_b *
                 (n_hyps * self.odim)).reshape([-1, 1])).reshape([-1])
        # select hypothesis scores
        s_new = paddle.index_select(s.reshape([-1]), vidx, 0)
        s_new = s_new.reshape([-1, 1]).repeat(1, self.odim).reshape(
            [n_bh, self.odim])
        # convert ids to BHS space (S: scoring_num)
        if scoring_idmap is not None:
            snum = self.scoring_num
            hyp_idx = (best_ids // self.odim +
                       (self.idx_b * n_hyps).reshape([-1, 1])).reshape([-1])
            label_ids = paddle.fmod(best_ids, self.odim).reshape([-1])
            score_idx = scoring_idmap[hyp_idx, label_ids]
            score_idx[score_idx == -1] = 0
            vidx = score_idx + hyp_idx * snum
        else:
            snum = self.odim
        # select forward probabilities
        r_new = paddle.index_select(r.reshape([-1, 2, n_bh * snum]), vidx,
                                    2).reshape([-1, 2, n_bh])
        return r_new, s_new, f_min, f_max

    def extend_prob(self, x):
        """Extend CTC prob.

        :param paddle.Tensor x: input label posterior sequences (B, T, O)
        """

        if self.x.shape[1] < x.shape[1]:  # self.x (2,T,B,O); x (B,T,O)
            # Pad the rest of posteriors in the batch
            # TODO(takaaki-hori): need a better way without for-loops
            xlens = [paddle.shape(x)[1]]
            for i, l in enumerate(xlens):
                if l < self.input_length:
                    x[i, l:, :] = self.logzero
                    x[i, l:, self.blank] = 0
            tmp_x = self.x
            xn = x.transpose([1, 0, 2])  # (B, T, O) -> (T, B, O)
            xb = xn[:, :, self.blank].unsqueeze(2).expand(-1, -1, self.odim)
            self.x = paddle.stack([xn, xb])  # (2, T, B, O)
            self.x[:, :tmp_x.shape[1], :, :] = tmp_x
            self.input_length = paddle.shape(x)[1]
            self.end_frames = paddle.to_tensor(xlens) - 1

    def extend_state(self, state):
        """Compute CTC prefix state.


        :param state    : CTC state
        :return ctc_state
        """

        if state is None:
            # nothing to do
            return state
        else:
            r_prev, s_prev, f_min_prev, f_max_prev = state

            r_prev_new = paddle.full(
                (self.input_length, 2),
                self.logzero,
                dtype=self.dtype, )
            start = max(r_prev.shape[0], 1)
            r_prev_new[0:start] = r_prev
            for t in range(start, self.input_length):
                r_prev_new[t, 1] = r_prev_new[t - 1, 1] + self.x[0, t, :,
                                                                 self.blank]

            return (r_prev_new, s_prev, f_min_prev, f_max_prev)


class CTCPrefixScore():
    """Compute CTC label sequence scores

    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    but extended to efficiently compute the probabilities of multiple labels
    simultaneously
    """

    def __init__(self, x, blank, eos, xp):
        self.xp = xp
        self.logzero = -10000000000.0
        self.blank = blank
        self.eos = eos
        self.input_length = len(x)
        self.x = x  # (T, O)

    def initial_state(self):
        """Obtain an initial CTC state

        :return: CTC state
        """
        # initial CTC state is made of a frame x 2 tensor that corresponds to
        # r_t^n(<sos>) and r_t^b(<sos>), where 0 and 1 of axis=1 represent
        # superscripts n and b (non-blank and blank), respectively.
        # r shape (T, 2)
        r = self.xp.full((self.input_length, 2), self.logzero, dtype=np.float32)
        r[0, 1] = self.x[0, self.blank]
        for i in six.moves.range(1, self.input_length):
            r[i, 1] = r[i - 1, 1] + self.x[i, self.blank]
        return r

    def __call__(self, y, cs, r_prev):
        """Compute CTC prefix scores for next labels

        :param y     : prefix label sequence
        :param cs    : array of next labels
        :param r_prev: previous CTC state
        :return ctc_scores, ctc_states
        """
        # initialize CTC states
        output_length = len(y) - 1  # ignore sos
        # new CTC states are prepared as a frame x (n or b) x n_labels tensor
        # that corresponds to r_t^n(h) and r_t^b(h).
        # r shape (T, 2, n_labels)
        r = self.xp.ndarray((self.input_length, 2, len(cs)), dtype=np.float32)
        xs = self.x[:, cs]
        if output_length == 0:
            r[0, 0] = xs[0]
            r[0, 1] = self.logzero
        else:
            # Although the code does not exactly follow Algorithm 2,
            # we don't have to change it because we can assume
            # r_t(h)=0 for t < |h| in CTC forward computation
            # (Note: we assume here that index t starts with 0).
            # The purpose of this difference is to reduce the number of for-loops.
            # https://github.com/espnet/espnet/pull/3655
            # where we start to accumulate r_t(h) from t=|h|
            # and iterate r_t(h) = (r_{t-1}(h) + ...) to T-1,
            # avoiding accumulating zeros for t=1~|h|-1.
            # Thus, we need to set r_{|h|-1}(h) = 0,
            # i.e., r[output_length-1] = logzero, for initialization.
            # This is just for reducing the computation.
            r[output_length - 1] = self.logzero

        # prepare forward probabilities for the last label
        r_sum = self.xp.logaddexp(r_prev[:, 0],
                                  r_prev[:, 1])  # log(r_t^n(g) + r_t^b(g))
        last = y[-1]
        if output_length > 0 and last in cs:
            log_phi = self.xp.ndarray(
                (self.input_length, len(cs)), dtype=np.float32)
            for i in six.moves.range(len(cs)):
                log_phi[:, i] = r_sum if cs[i] != last else r_prev[:, 1]
        else:
            log_phi = r_sum

        # compute forward probabilities log(r_t^n(h)), log(r_t^b(h)),
        # and log prefix probabilities log(psi)
        start = max(output_length, 1)
        log_psi = r[start - 1, 0]
        for t in six.moves.range(start, self.input_length):
            r[t, 0] = self.xp.logaddexp(r[t - 1, 0], log_phi[t - 1]) + xs[t]
            r[t, 1] = (self.xp.logaddexp(r[t - 1, 0], r[t - 1, 1]) +
                       self.x[t, self.blank])
            log_psi = self.xp.logaddexp(log_psi, log_phi[t - 1] + xs[t])

        # get P(...eos|X) that ends with the prefix itself
        eos_pos = self.xp.where(cs == self.eos)[0]
        if len(eos_pos) > 0:
            log_psi[eos_pos] = r_sum[-1]  # log(r_T^n(g) + r_T^b(g))

        # exclude blank probs
        blank_pos = self.xp.where(cs == self.blank)[0]
        if len(blank_pos) > 0:
            log_psi[blank_pos] = self.logzero

        # return the log prefix probability and CTC states, where the label axis
        # of the CTC states is moved to the first axis to slice it easily
        # log_psi shape (n_labels,), state shape (n_labels, T, 2)
        return log_psi, self.xp.rollaxis(r, 2)
