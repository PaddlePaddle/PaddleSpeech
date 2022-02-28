"""Decoders and output normalization for CTC.

Authors
 * Mirco Ravanelli 2020
 * Aku Rouhe 2020
 * Sung-Lin Yeh 2020
"""
import paddle
from itertools import groupby
from speechbrain.dataio.dataio import length_to_mask


class CTCPrefixScorer:
    """This class implements the CTC prefix scorer of Algorithm 2 in
    reference: https://www.merl.com/publications/docs/TR2017-190.pdf.
    Official implementation: https://github.com/espnet/espnet/blob/master/espnet/nets/ctc_prefix_score.py

    Arguments
    ---------
    x : paddle.Tensor
        The encoder states.
    enc_lens : paddle.Tensor
        The actual length of each enc_states sequence.
    batch_size : int
        The size of the batch.
    beam_size : int
        The width of beam.
    blank_index : int
        The index of the blank token.
    eos_index : int
        The index of the end-of-sequence (eos) token.
    ctc_window_size: int
        Compute the ctc scores over the time frames using windowing based on attention peaks.
        If 0, no windowing applied.
    """

    def __init__(
        self,
        x,
        enc_lens,
        batch_size,
        beam_size,
        blank_index,
        eos_index,
        ctc_window_size=0,
    ):
        self.blank_index = blank_index
        self.eos_index = eos_index
        self.max_enc_len = x.size(1)
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.vocab_size = x.size(-1)
        self.device = x.device
        self.minus_inf = -1e20
        self.last_frame_index = enc_lens - 1
        self.ctc_window_size = ctc_window_size

        # mask frames > enc_lens
        mask = 1 - length_to_mask(enc_lens)
        mask = mask.unsqueeze(-1).expand(-1, -1, x.size(-1)).eq(1)
        x.masked_fill_(mask, self.minus_inf)
        x[:, :, 0] = x[:, :, 0].masked_fill_(mask[:, :, 0], 0)

        # dim=0: xnb, nonblank posteriors, dim=1: xb, blank posteriors
        xnb = x.transpose(0, 1)
        xb = (
            xnb[:, :, self.blank_index]
            .unsqueeze(2)
            .expand(-1, -1, self.vocab_size)
        )

        # (2, L, batch_size * beam_size, vocab_size)
        self.x = paddle.stack([xnb, xb])

        # The first index of each sentence.
        self.beam_offset = (
            paddle.arange(batch_size, device=self.device) * self.beam_size
        )
        # The first index of each candidates.
        self.cand_offset = (
            paddle.arange(batch_size, device=self.device) * self.vocab_size
        )

    def forward_step(self, g, state, candidates=None, attn=None):
        """This method if one step of forwarding operation
        for the prefix ctc scorer.

        Arguments
        ---------
        g : paddle.Tensor
            The tensor of prefix label sequences, h = g + c.
        state : tuple
            Previous ctc states.
        candidates : paddle.Tensor
            (batch_size * beam_size, ctc_beam_size), The topk candidates for rescoring.
            The ctc_beam_size is set as 2 * beam_size. If given, performing partial ctc scoring.
        """

        prefix_length = g.size(1)
        last_char = [gi[-1] for gi in g] if prefix_length > 0 else [0] * len(g)
        self.num_candidates = (
            self.vocab_size if candidates is None else candidates.size(-1)
        )
        if state is None:
            # r_prev: (L, 2, batch_size * beam_size)
            r_prev = paddle.full(
                (self.max_enc_len, 2, self.batch_size, self.beam_size),
                self.minus_inf,
                device=self.device,
            )

            # Accumulate blank posteriors at each step
            r_prev[:, 1] = paddle.cumsum(
                self.x[0, :, :, self.blank_index], 0
            ).unsqueeze(2)
            r_prev = r_prev.view(-1, 2, self.batch_size * self.beam_size)
            psi_prev = 0.0
        else:
            r_prev, psi_prev = state

        # for partial search
        if candidates is not None:
            scoring_table = paddle.full(
                (self.batch_size * self.beam_size, self.vocab_size),
                -1,
                dtype=paddle.long,
                device=self.device,
            )
            # Assign indices of candidates to their positions in the table
            col_index = paddle.arange(
                self.batch_size * self.beam_size, device=self.device
            ).unsqueeze(1)
            scoring_table[col_index, candidates] = paddle.arange(
                self.num_candidates, device=self.device
            )
            # Select candidates indices for scoring
            scoring_index = (
                candidates
                + self.cand_offset.unsqueeze(1)
                .repeat(1, self.beam_size)
                .view(-1, 1)
            ).view(-1)
            x_inflate = paddle.index_select(
                self.x.view(2, -1, self.batch_size * self.vocab_size),
                2,
                scoring_index,
            ).view(2, -1, self.batch_size * self.beam_size, self.num_candidates)
        # for full search
        else:
            scoring_table = None
            x_inflate = (
                self.x.unsqueeze(3)
                .repeat(1, 1, 1, self.beam_size, 1)
                .view(
                    2, -1, self.batch_size * self.beam_size, self.num_candidates
                )
            )

        # Prepare forward probs
        r = paddle.full(
            (
                self.max_enc_len,
                2,
                self.batch_size * self.beam_size,
                self.num_candidates,
            ),
            self.minus_inf,
            device=self.device,
        )
        r.fill_(self.minus_inf)

        # (Alg.2-6)
        if prefix_length == 0:
            r[0, 0] = x_inflate[0, 0]
        # (Alg.2-10): phi = prev_nonblank + prev_blank = r_t-1^nb(g) + r_t-1^b(g)
        r_sum = paddle.logsumexp(r_prev, 1)
        phi = r_sum.unsqueeze(2).repeat(1, 1, self.num_candidates)

        # (Alg.2-10): if last token of prefix g in candidates, phi = prev_b + 0
        if candidates is not None:
            for i in range(self.batch_size * self.beam_size):
                pos = scoring_table[i, last_char[i]]
                if pos != -1:
                    phi[:, i, pos] = r_prev[:, 1, i]
        else:
            for i in range(self.batch_size * self.beam_size):
                phi[:, i, last_char[i]] = r_prev[:, 1, i]

        # Start, end frames for scoring (|g| < |h|).
        # Scoring based on attn peak if ctc_window_size > 0
        if self.ctc_window_size == 0 or attn is None:
            start = max(1, prefix_length)
            end = self.max_enc_len
        else:
            _, attn_peak = paddle.max(attn, dim=1)
            max_frame = paddle.max(attn_peak).item() + self.ctc_window_size
            min_frame = paddle.min(attn_peak).item() - self.ctc_window_size
            start = max(max(1, prefix_length), int(min_frame))
            end = min(self.max_enc_len, int(max_frame))

        # Compute forward prob log(r_t^nb(h)) and log(r_t^b(h)):
        for t in range(start, end):
            # (Alg.2-11): dim=0, p(h|cur step is nonblank) = [p(prev step=y) + phi] * p(c)
            rnb_prev = r[t - 1, 0]
            # (Alg.2-12): dim=1, p(h|cur step is blank) = [p(prev step is blank) + p(prev step is nonblank)] * p(blank)
            rb_prev = r[t - 1, 1]
            r_ = paddle.stack([rnb_prev, phi[t - 1], rnb_prev, rb_prev]).view(
                2, 2, self.batch_size * self.beam_size, self.num_candidates
            )
            r[t] = paddle.logsumexp(r_, 1) + x_inflate[:, t]

        # Compute the predix prob, psi
        psi_init = r[start - 1, 0].unsqueeze(0)
        # phi is prob at t-1 step, shift one frame and add it to the current prob p(c)
        phix = paddle.cat((phi[0].unsqueeze(0), phi[:-1]), dim=0) + x_inflate[0]
        # (Alg.2-13): psi = psi + phi * p(c)
        if candidates is not None:
            psi = paddle.full(
                (self.batch_size * self.beam_size, self.vocab_size),
                self.minus_inf,
                device=self.device,
            )
            psi_ = paddle.logsumexp(
                paddle.cat((phix[start:end], psi_init), dim=0), dim=0
            )
            # only assign prob to candidates
            for i in range(self.batch_size * self.beam_size):
                psi[i, candidates[i]] = psi_[i]
        else:
            psi = paddle.logsumexp(
                paddle.cat((phix[start:end], psi_init), dim=0), dim=0
            )

        # (Alg.2-3): if c = <eos>, psi = log(r_T^n(g) + r_T^b(g)), where T is the length of max frames
        for i in range(self.batch_size * self.beam_size):
            psi[i, self.eos_index] = r_sum[
                self.last_frame_index[i // self.beam_size], i
            ]

        # Exclude blank probs for joint scoring
        psi[:, self.blank_index] = self.minus_inf

        return psi - psi_prev, (r, psi, scoring_table)

    def permute_mem(self, memory, index):
        """This method permutes the CTC model memory
        to synchronize the memory index with the current output.

        Arguments
        ---------
        memory : No limit
            The memory variable to be permuted.
        index : paddle.Tensor
            The index of the previous path.

        Return
        ------
        The variable of the memory being permuted.

        """
        r, psi, scoring_table = memory
        # The index of top-K vocab came from in (t-1) timesteps.
        best_index = (
            index
            + (self.beam_offset.unsqueeze(1).expand_as(index) * self.vocab_size)
        ).view(-1)
        # synchronize forward prob
        psi = paddle.index_select(psi.view(-1), dim=0, index=best_index)
        psi = (
            psi.view(-1, 1)
            .repeat(1, self.vocab_size)
            .view(self.batch_size * self.beam_size, self.vocab_size)
        )

        # synchronize ctc states
        if scoring_table is not None:
            effective_index = (
                index // self.vocab_size + self.beam_offset.view(-1, 1)
            ).view(-1)
            selected_vocab = (index % self.vocab_size).view(-1)
            score_index = scoring_table[effective_index, selected_vocab]
            score_index[score_index == -1] = 0
            best_index = score_index + effective_index * self.num_candidates

        r = paddle.index_select(
            r.view(
                -1, 2, self.batch_size * self.beam_size * self.num_candidates
            ),
            dim=-1,
            index=best_index,
        )
        r = r.view(-1, 2, self.batch_size * self.beam_size)

        return r, psi


def filter_ctc_output(string_pred, blank_id=-1):
    """Apply CTC output merge and filter rules.

    Removes the blank symbol and output repetitions.

    Arguments
    ---------
    string_pred : list
        A list containing the output strings/ints predicted by the CTC system.
    blank_id : int, string
        The id of the blank.

    Returns
    -------
    list
        The output predicted by CTC without the blank symbol and
        the repetitions.

    Example
    -------
    >>> string_pred = ['a','a','blank','b','b','blank','c']
    >>> string_out = filter_ctc_output(string_pred, blank_id='blank')
    >>> print(string_out)
    ['a', 'b', 'c']
    """

    if isinstance(string_pred, list):
        # Filter the repetitions
        string_out = [
            v
            for i, v in enumerate(string_pred)
            if i == 0 or v != string_pred[i - 1]
        ]

        # Remove duplicates
        string_out = [i[0] for i in groupby(string_out)]

        # Filter the blank symbol
        string_out = list(filter(lambda elem: elem != blank_id, string_out))
    else:
        raise ValueError("filter_ctc_out can only filter python lists")
    return string_out


def ctc_greedy_decode(probabilities, seq_lens, blank_id=-1):
    """Greedy decode a batch of probabilities and apply CTC rules.

    Arguments
    ---------
    probabilities : paddle.tensor
        Output probabilities (or log-probabilities) from the network with shape
        [batch, probabilities, time]
    seq_lens : paddle.tensor
        Relative true sequence lengths (to deal with padded inputs),
        the longest sequence has length 1.0, others a value between zero and one
        shape [batch, lengths].
    blank_id : int, string
        The blank symbol/index. Default: -1. If a negative number is given,
        it is assumed to mean counting down from the maximum possible index,
        so that -1 refers to the maximum possible index.

    Returns
    -------
    list
        Outputs as Python list of lists, with "ragged" dimensions; padding
        has been removed.

    Example
    -------
    >>> import paddle
    >>> probs = paddle.tensor([[[0.3, 0.7], [0.0, 0.0]],
    ...                       [[0.2, 0.8], [0.9, 0.1]]])
    >>> lens = paddle.tensor([0.51, 1.0])
    >>> blank_id = 0
    >>> ctc_greedy_decode(probs, lens, blank_id)
    [[1], [1]]
    """
    if isinstance(blank_id, int) and blank_id < 0:
        blank_id = probabilities.shape[-1] + blank_id
    batch_max_len = probabilities.shape[1]
    batch_outputs = []
    for seq, seq_len in zip(probabilities, seq_lens):
        actual_size = int(paddle.round(seq_len * batch_max_len))
        scores, predictions = paddle.max(seq.narrow(0, 0, actual_size), dim=1)
        out = filter_ctc_output(predictions.tolist(), blank_id=blank_id)
        batch_outputs.append(out)
    return batch_outputs
