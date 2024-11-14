# Copyright 2021 Mobvoi Inc. All Rights Reserved.
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
# Modified from wenet(https://github.com/wenet-e2e/wenet)
"""U2 ASR Model
Unified Streaming and Non-streaming Two-pass End-to-end Model for Speech Recognition
(https://arxiv.org/pdf/2012.05481.pdf)
"""
import sys
import time
from collections import defaultdict
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import paddle
from paddle import jit
from paddle import nn

from paddlespeech.audio.utils.tensor_utils import add_sos_eos
from paddlespeech.audio.utils.tensor_utils import pad_sequence
from paddlespeech.audio.utils.tensor_utils import reverse_pad_list
from paddlespeech.audio.utils.tensor_utils import st_reverse_pad_list
from paddlespeech.audio.utils.tensor_utils import th_accuracy
from paddlespeech.s2t.decoders.scorers.ctc import CTCPrefixScorer
from paddlespeech.s2t.frontend.utility import IGNORE_ID
from paddlespeech.s2t.frontend.utility import load_cmvn
from paddlespeech.s2t.models.asr_interface import ASRInterface
from paddlespeech.s2t.modules.cmvn import GlobalCMVN
from paddlespeech.s2t.modules.ctc import CTCDecoderBase
from paddlespeech.s2t.modules.decoder import BiTransformerDecoder
from paddlespeech.s2t.modules.decoder import TransformerDecoder
from paddlespeech.s2t.modules.encoder import ConformerEncoder
from paddlespeech.s2t.modules.encoder import SqueezeformerEncoder
from paddlespeech.s2t.modules.encoder import TransformerEncoder
from paddlespeech.s2t.modules.initializer import DefaultInitializerContext
from paddlespeech.s2t.modules.loss import LabelSmoothingLoss
from paddlespeech.s2t.modules.mask import make_pad_mask
from paddlespeech.s2t.modules.mask import mask_finished_preds
from paddlespeech.s2t.modules.mask import mask_finished_scores
from paddlespeech.s2t.modules.mask import subsequent_mask
from paddlespeech.s2t.utils import checkpoint
from paddlespeech.s2t.utils import layer_tools
from paddlespeech.s2t.utils.ctc_utils import remove_duplicates_and_blank
from paddlespeech.s2t.utils.log import Log
from paddlespeech.s2t.utils.utility import log_add
from paddlespeech.s2t.utils.utility import UpdateConfig

__all__ = ["U2Model", "U2InferModel"]

logger = Log(__name__).getlog()


class U2BaseModel(ASRInterface, nn.Layer):
    """CTC-Attention hybrid Encoder-Decoder model"""

    def __init__(self,
                 vocab_size: int,
                 encoder: TransformerEncoder,
                 decoder: TransformerDecoder,
                 ctc: CTCDecoderBase,
                 ctc_weight: float=0.5,
                 ignore_id: int=IGNORE_ID,
                 reverse_weight: float=0.0,
                 lsm_weight: float=0.0,
                 length_normalized_loss: bool=False,
                 **kwargs):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        nn.Layer.__init__(self)

        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight

        self.encoder = encoder
        self.decoder = decoder
        self.ctc = ctc
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss, )

    def forward(
            self,
            speech: paddle.Tensor,
            speech_lengths: paddle.Tensor,
            text: paddle.Tensor,
            text_lengths: paddle.Tensor,
    ) -> Tuple[Optional[paddle.Tensor], Optional[paddle.Tensor], Optional[
            paddle.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        Returns:
            total_loss, attention_loss, ctc_loss
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)
        # 1. Encoder
        start = time.time()
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_time = time.time() - start
        #logger.debug(f"encoder time: {encoder_time}")
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)  #[B, 1, T] -> [B]

        # 2a. Attention-decoder branch
        loss_att = None
        if self.ctc_weight != 1.0:
            start = time.time()
            loss_att, acc_att = self._calc_att_loss(encoder_out, encoder_mask,
                                                    text, text_lengths,
                                                    self.reverse_weight)
            decoder_time = time.time() - start
            #logger.debug(f"decoder time: {decoder_time}")

        # 2b. CTC branch
        loss_ctc = None
        if self.ctc_weight != 0.0:
            start = time.time()
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, text,
                                text_lengths)
            ctc_time = time.time() - start
            #logger.debug(f"ctc time: {ctc_time}")
        if loss_ctc is None:
            loss = loss_att
        elif loss_att is None:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
        return loss, loss_att, loss_ctc

    def _calc_att_loss(self,
                       encoder_out: paddle.Tensor,
                       encoder_mask: paddle.Tensor,
                       ys_pad: paddle.Tensor,
                       ys_pad_lens: paddle.Tensor,
                       reverse_weight: float) -> Tuple[paddle.Tensor, float]:
        """Calc attention loss.

        Args:
            encoder_out (paddle.Tensor): [B, Tmax, D]
            encoder_mask (paddle.Tensor): [B, 1, Tmax]
            ys_pad (paddle.Tensor): [B, Umax]
            ys_pad_lens (paddle.Tensor): [B]
            reverse_weight (float): reverse decoder weight.

        Returns:
            Tuple[paddle.Tensor, float]: attention_loss, accuracy rate
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos,
                                            self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        r_ys_pad = reverse_pad_list(ys_pad, ys_pad_lens, float(self.ignore_id))
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, self.sos, self.eos,
                                                self.ignore_id)
        # 1. Forward decoder
        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, ys_in_pad, ys_in_lens, r_ys_in_pad,
            reverse_weight)

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        r_loss_att = paddle.to_tensor(0.0)
        if reverse_weight > 0.0:
            r_loss_att = self.criterion_att(r_decoder_out, r_ys_out_pad)
        loss_att = loss_att * (1 - reverse_weight) + r_loss_att * reverse_weight
        acc_att = th_accuracy(
            decoder_out.reshape([-1, self.vocab_size]),
            ys_out_pad,
            ignore_label=self.ignore_id, )
        return loss_att, acc_att

    def _forward_encoder(
            self,
            speech: paddle.Tensor,
            speech_lengths: paddle.Tensor,
            decoding_chunk_size: int=-1,
            num_decoding_left_chunks: int=-1,
            simulate_streaming: bool=False,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Encoder pass.

        Args:
            speech (paddle.Tensor): [B, Tmax, D]
            speech_lengths (paddle.Tensor): [B]
            decoding_chunk_size (int, optional): chuck size. Defaults to -1.
            num_decoding_left_chunks (int, optional): nums chunks. Defaults to -1.
            simulate_streaming (bool, optional): streaming or not. Defaults to False.

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]:
                encoder hiddens (B, Tmax, D),
                encoder hiddens mask (B, 1, Tmax).
        """
        # Let's assume B = batch_size
        # 1. Encoder
        if simulate_streaming and decoding_chunk_size > 0:
            encoder_out, encoder_mask = self.encoder.forward_chunk_by_chunk(
                speech,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        else:
            encoder_out, encoder_mask = self.encoder(
                speech,
                speech_lengths,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        return encoder_out, encoder_mask

    def recognize(
            self,
            speech: paddle.Tensor,
            speech_lengths: paddle.Tensor,
            beam_size: int=10,
            decoding_chunk_size: int=-1,
            num_decoding_left_chunks: int=-1,
            simulate_streaming: bool=False, ) -> paddle.Tensor:
        """ Apply beam search on attention decoder
        Args:
            speech (paddle.Tensor): (batch, max_len, feat_dim)
            speech_length (paddle.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
        Returns:
            paddle.Tensor: decoding result, (batch, max_result_len)
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        device = speech.place
        batch_size = speech.shape[0]

        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.shape[1]
        encoder_dim = encoder_out.shape[2]
        running_size = batch_size * beam_size
        encoder_out = encoder_out.unsqueeze(1).repeat(
            1, beam_size, 1, 1).reshape(
                [running_size, maxlen,
                 encoder_dim])  # (B*N, maxlen, encoder_dim)
        encoder_mask = encoder_mask.unsqueeze(1).repeat(
            1, beam_size, 1, 1).reshape([running_size, 1,
                                         maxlen])  # (B*N, 1, max_len)

        hyps = paddle.ones(
            [running_size, 1], dtype=paddle.long).fill_(self.sos)  # (B*N, 1)
        # log scale score
        scores = paddle.to_tensor(
            [0.0] + [-float('inf')] * (beam_size - 1), dtype=paddle.float)
        scores = scores.to(device).repeat(batch_size).unsqueeze(1).to(
            device)  # (B*N, 1)
        end_flag = paddle.zeros_like(scores, dtype=paddle.bool)  # (B*N, 1)
        cache: Optional[List[paddle.Tensor]] = None
        # 2. Decoder forward step by step
        for i in range(1, maxlen + 1):
            # Stop if all batch and all beam produce eos
            if end_flag.sum() == running_size:
                break

            # 2.1 Forward decoder step
            hyps_mask = subsequent_mask(i).unsqueeze(0).repeat(
                running_size, 1, 1).to(device)  # (B*N, i, i)
            # logp: (B*N, vocab)
            logp, cache = self.decoder.forward_one_step(
                encoder_out, encoder_mask, hyps, hyps_mask, cache)
            # 2.2 First beam prune: select topk best prob at current time
            top_k_logp, top_k_index = logp.topk(beam_size)  # (B*N, N)
            top_k_logp = mask_finished_scores(top_k_logp, end_flag)
            top_k_index = mask_finished_preds(top_k_index, end_flag, self.eos)

            # 2.3 Seconde beam prune: select topk score with history
            scores = scores + top_k_logp  # (B*N, N), broadcast add
            scores = scores.reshape(
                [batch_size, beam_size * beam_size])  # (B, N*N)
            scores, offset_k_index = scores.topk(k=beam_size)  # (B, N)
            scores = scores.reshape([-1, 1])  # (B*N, 1)

            # 2.4. Compute base index in top_k_index,
            # regard top_k_index as (B*N*N),regard offset_k_index as (B*N),
            # then find offset_k_index in top_k_index
            base_k_index = paddle.arange(batch_size).reshape([-1, 1]).repeat(
                1, beam_size)  # (B, N)
            base_k_index = base_k_index * beam_size * beam_size
            best_k_index = base_k_index.reshape([-1]) + offset_k_index.reshape(
                [-1])  # (B*N)

            # 2.5 Update best hyps
            best_k_pred = paddle.index_select(
                top_k_index.reshape([-1]), index=best_k_index, axis=0)  # (B*N)
            best_hyps_index = best_k_index // beam_size
            last_best_k_hyps = paddle.index_select(
                hyps, index=best_hyps_index, axis=0)  # (B*N, i)
            hyps = paddle.cat(
                (last_best_k_hyps, best_k_pred.reshape([-1, 1])),
                dim=1)  # (B*N, i+1)

            # 2.6 Update end flag
            end_flag = paddle.equal(hyps[:, -1], self.eos).reshape([-1, 1])

        # 3. Select best of best
        scores = scores.reshape([batch_size, beam_size])
        # TODO: length normalization
        best_index = paddle.argmax(scores, axis=-1).long()  # (B)
        best_hyps_index = best_index + paddle.arange(
            batch_size, dtype=paddle.long) * beam_size
        best_hyps = paddle.index_select(hyps, index=best_hyps_index, axis=0)
        best_hyps = best_hyps[:, 1:]
        return best_hyps

    def ctc_greedy_search(
            self,
            speech: paddle.Tensor,
            speech_lengths: paddle.Tensor,
            decoding_chunk_size: int=-1,
            num_decoding_left_chunks: int=-1,
            simulate_streaming: bool=False, ) -> List[List[int]]:
        """ Apply CTC greedy search
        Args:
            speech (paddle.Tensor): (batch, max_len, feat_dim)
            speech_length (paddle.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
        Returns:
            List[List[int]]: best path result
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        batch_size = speech.shape[0]

        # Let's assume B = batch_size
        # encoder_out: (B, maxlen, encoder_dim)
        # encoder_mask: (B, 1, Tmax)

        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks, simulate_streaming)
        maxlen = encoder_out.shape[1]
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        ctc_probs = self.ctc.log_softmax(encoder_out)  # (B, maxlen, vocab_size)

        topk_prob, topk_index = ctc_probs.topk(1, axis=2)  # (B, maxlen, 1)
        topk_index = topk_index.reshape([batch_size, maxlen])  # (B, maxlen)
        pad_mask = make_pad_mask(encoder_out_lens)  # (B, maxlen)
        topk_index = topk_index.masked_fill_(pad_mask, self.eos)  # (B, maxlen)

        hyps = [hyp.tolist() for hyp in topk_index]
        hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]
        return hyps

    def _ctc_prefix_beam_search(
            self,
            speech: paddle.Tensor,
            speech_lengths: paddle.Tensor,
            beam_size: int,
            decoding_chunk_size: int=-1,
            num_decoding_left_chunks: int=-1,
            simulate_streaming: bool=False,
            blank_id: int=0, ) -> Tuple[List[Tuple[int, float]], paddle.Tensor]:
        """ CTC prefix beam search inner implementation
        Args:
            speech (paddle.Tensor): (batch, max_len, feat_dim)
            speech_length (paddle.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
        Returns:
            List[Tuple[int, float]]: nbest results, (N,1), (text, likelihood)
            paddle.Tensor: encoder output, (1, max_len, encoder_dim),
                it will be used for rescoring in attention rescoring mode
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        batch_size = speech.shape[0]
        # For CTC prefix beam search, we only support batch_size=1
        assert batch_size == 1

        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder forward and get CTC score
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.shape[1]
        ctc_probs = self.ctc.log_softmax(encoder_out)  # (1, maxlen, vocab_size)
        ctc_probs = ctc_probs.squeeze(0)

        # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
        # blank_ending_score and  none_blank_ending_score in ln domain
        cur_hyps = [(tuple(), (0.0, -float('inf')))]
        # 2. CTC beam search step by step
        for t in range(0, maxlen):
            logp = ctc_probs[t]  # (vocab_size,)
            # key: prefix, value (pb, pnb), default value(-inf, -inf)
            next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
            # 2.1 First beam prune: select topk best
            top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
            for s in top_k_index:
                s = s.item()
                ps = logp[s].item()
                for prefix, (pb, pnb) in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == blank_id:  # blank
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pb = log_add([n_pb, pb + ps, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                    elif s == last:
                        #  Update *ss -> *s;
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pnb = log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
                    else:
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)

            # 2.2 Second beam prune
            next_hyps = sorted(
                next_hyps.items(),
                key=lambda x: log_add(list(x[1])),
                reverse=True)
            cur_hyps = next_hyps[:beam_size]

        hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
        return hyps, encoder_out

    def ctc_prefix_beam_search(
            self,
            speech: paddle.Tensor,
            speech_lengths: paddle.Tensor,
            beam_size: int,
            decoding_chunk_size: int=-1,
            num_decoding_left_chunks: int=-1,
            simulate_streaming: bool=False, ) -> List[int]:
        """ Apply CTC prefix beam search
        Args:
            speech (paddle.Tensor): (batch, max_len, feat_dim)
            speech_length (paddle.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
        Returns:
            List[int]: CTC prefix beam search nbest results
        """
        hyps, _ = self._ctc_prefix_beam_search(
            speech, speech_lengths, beam_size, decoding_chunk_size,
            num_decoding_left_chunks, simulate_streaming)
        return hyps[0][0]

    def attention_rescoring(self,
                            speech: paddle.Tensor,
                            speech_lengths: paddle.Tensor,
                            beam_size: int,
                            decoding_chunk_size: int=-1,
                            num_decoding_left_chunks: int=-1,
                            ctc_weight: float=0.0,
                            simulate_streaming: bool=False,
                            reverse_weight: float=0.0) -> List[int]:
        """ Apply attention rescoring decoding, CTC prefix beam search
            is applied first to get nbest, then we resoring the nbest on
            attention decoder with corresponding encoder out
        Args:
            speech (paddle.Tensor): (batch, max_len, feat_dim)
            speech_length (paddle.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
            reverse_weight (float): reverse deocder weight.
        Returns:
            List[int]: Attention rescoring result
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        if reverse_weight > 0.0:
            # decoder should be a bitransformer decoder if reverse_weight > 0.0
            assert hasattr(self.decoder, 'right_decoder')
        device = speech.place
        batch_size = speech.shape[0]
        # For attention rescoring we only support batch_size=1
        assert batch_size == 1

        # len(hyps) = beam_size, encoder_out: (1, maxlen, encoder_dim)
        hyps, encoder_out = self._ctc_prefix_beam_search(
            speech, speech_lengths, beam_size, decoding_chunk_size,
            num_decoding_left_chunks, simulate_streaming)
        assert len(hyps) == beam_size

        hyp_list = []
        for hyp in hyps:
            hyp_content = hyp[0]
            # Prevent the hyp is empty
            if len(hyp_content) == 0:
                hyp_content = (self.ctc.blank_id, )
            hyp_content = paddle.to_tensor(
                hyp_content, place=device, dtype=paddle.long)
            hyp_list.append(hyp_content)
        hyps_pad = pad_sequence(hyp_list, True, self.ignore_id)
        hyps_lens = paddle.to_tensor(
            [len(hyp[0]) for hyp in hyps], place=device,
            dtype=paddle.long)  # (beam_size,)
        hyps_pad, _ = add_sos_eos(hyps_pad, self.sos, self.eos, self.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at beginning
        logger.debug(
            f"hyps pad: {hyps_pad} {self.sos} {self.eos} {self.ignore_id}")

        # ctc score in ln domain
        # (beam_size, max_hyps_len, vocab_size)
        decoder_out, r_decoder_out = self.forward_attention_decoder(
            hyps_pad, hyps_lens, encoder_out, reverse_weight)

        decoder_out = decoder_out.numpy()
        # r_decoder_out will be 0.0, if reverse_weight is 0.0 or decoder is a
        # conventional transformer decoder.
        r_decoder_out = r_decoder_out.numpy()

        # Only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0
        # hyps is List[(Text=List[int], Score=float)], len(hyps)=beam_size
        for i, hyp in enumerate(hyps):
            score = 0.0
            for j, w in enumerate(hyp[0]):
                score += decoder_out[i][j][w]
            # last decoder output token is `eos`, for laste decoder input token.
            score += decoder_out[i][len(hyp[0])][self.eos]

            logger.debug(
                f"hyp {i} len {len(hyp[0])} l2r score: {score} ctc_score: {hyp[1]} reverse_weight: {reverse_weight}"
            )

            if reverse_weight > 0:
                r_score = 0.0
                for j, w in enumerate(hyp[0]):
                    r_score += r_decoder_out[i][len(hyp[0]) - j - 1][w]
                r_score += r_decoder_out[i][len(hyp[0])][self.eos]

                logger.debug(
                    f"hyp {i} len {len(hyp[0])} r2l score: {r_score} ctc_score: {hyp[1]} reverse_weight: {reverse_weight}"
                )

                score = score * (1 - reverse_weight) + r_score * reverse_weight
            # add ctc score (which in ln domain)
            score += hyp[1] * ctc_weight
            if score > best_score:
                best_score = score
                best_index = i

        logger.debug(f"result: {hyps[best_index]}")
        return hyps[best_index][0]

    @jit.to_static(property=True)
    def subsampling_rate(self) -> int:
        """ Export interface for c++ call, return subsampling_rate of the
            model
        """
        return self.encoder.embed.subsampling_rate

    @jit.to_static(property=True)
    def right_context(self) -> int:
        """ Export interface for c++ call, return right_context of the model
        """
        return self.encoder.embed.right_context

    @jit.to_static(property=True)
    def sos_symbol(self) -> int:
        """ Export interface for c++ call, return sos symbol id of the model
        """
        return self.sos

    @jit.to_static(property=True)
    def eos_symbol(self) -> int:
        """ Export interface for c++ call, return eos symbol id of the model
        """
        return self.eos

    @jit.to_static(property=True)
    def is_bidirectional_decoder(self) -> bool:
        """
        Returns:
            paddle.Tensor: decoder output
        """
        if hasattr(self.decoder, 'right_decoder'):
            return True
        else:
            return False

    # @jit.to_static
    def forward_encoder_chunk(
            self,
            xs: paddle.Tensor,
            offset: int,
            required_cache_size: int,
            att_cache: paddle.Tensor=paddle.zeros([0, 0, 0, 0]),
            cnn_cache: paddle.Tensor=paddle.zeros([0, 0, 0, 0])
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """ Export interface for c++ call, give input chunk xs, and return
            output from time 0 to current chunk.

        Args:
            xs (paddle.Tensor): chunk input, with shape (b=1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            att_cache (paddle.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
                `d_k * 2` for att key & value. 
            cnn_cache (paddle.Tensor): cache tensor for cnn_module in conformer,
                (elayers, b=1, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`. 
                
        Returns:
            paddle.Tensor: output of current input xs,
                with shape (b=1, chunk_size, hidden-dim).
            paddle.Tensor: new attention cache required for next chunk, with
                dynamic shape (elayers, head, T(?), d_k * 2)
                depending on required_cache_size.
            paddle.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.
        """
        return self.encoder.forward_chunk(xs, offset, required_cache_size,
                                          att_cache, cnn_cache)

    # @jit.to_static
    def ctc_activation(self, xs: paddle.Tensor) -> paddle.Tensor:
        """ Export interface for c++ call, apply linear transform and log
            softmax before ctc
        Args:
            xs (paddle.Tensor): encoder output, (B, T, D)
        Returns:
            paddle.Tensor: activation before ctc. (B, Tmax, odim)
        """
        return self.ctc.log_softmax(xs)

    # @jit.to_static
    def forward_attention_decoder(self,
                                  hyps: paddle.Tensor,
                                  hyps_lens: paddle.Tensor,
                                  encoder_out: paddle.Tensor,
                                  reverse_weight: float=0.0) -> paddle.Tensor:
        """ Export interface for c++ call, forward decoder with multiple
            hypothesis from ctc prefix beam search and one encoder output
        Args:
            hyps (paddle.Tensor): hyps from ctc prefix beam search, already
                pad sos at the beginning, (B, T)
            hyps_lens (paddle.Tensor): length of each hyp in hyps, (B)
            encoder_out (paddle.Tensor): corresponding encoder output, (B=1, T, D)
        Returns:
            paddle.Tensor: decoder output, (B, L)
        """
        assert encoder_out.shape[0] == 1
        num_hyps = hyps.shape[0]
        assert hyps_lens.shape[0] == num_hyps
        encoder_out = encoder_out.repeat(num_hyps, 1, 1)
        # (B, 1, T)
        encoder_mask = paddle.ones(
            [num_hyps, 1, encoder_out.shape[1]], dtype=paddle.bool)

        # input for right to left decoder
        # this hyps_lens has count <sos> token, we need minus it.
        r_hyps_lens = hyps_lens - 1
        # this hyps has included <sos> token, so it should be
        # convert the original hyps.
        r_hyps = hyps[:, 1:]
        # (num_hyps, max_hyps_len, vocab_size)

        r_hyps = st_reverse_pad_list(r_hyps, r_hyps_lens, self.sos, self.eos)

        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps, hyps_lens, r_hyps, reverse_weight)
        decoder_out = paddle.nn.functional.log_softmax(decoder_out, axis=-1)
        r_decoder_out = paddle.nn.functional.log_softmax(r_decoder_out, axis=-1)
        return decoder_out, r_decoder_out

    @paddle.no_grad()
    def decode(self,
               feats: paddle.Tensor,
               feats_lengths: paddle.Tensor,
               text_feature: Dict[str, int],
               decoding_method: str,
               beam_size: int,
               ctc_weight: float=0.0,
               decoding_chunk_size: int=-1,
               num_decoding_left_chunks: int=-1,
               simulate_streaming: bool=False,
               reverse_weight: float=0.0):
        """u2 decoding.

        Args:
            feats (Tensor): audio features, (B, T, D)
            feats_lengths (Tensor): (B)
            text_feature (TextFeaturizer): text feature object.
            decoding_method (str): decoding mode, e.g.
                    'attention', 'ctc_greedy_search',
                    'ctc_prefix_beam_search', 'attention_rescoring'
            beam_size (int): beam size for search
            ctc_weight (float, optional): ctc weight for attention rescoring decode mode. Defaults to 0.0.
            decoding_chunk_size (int, optional): decoding chunk size. Defaults to -1.
                    <0: for decoding, use full chunk.
                    >0: for decoding, use fixed chunk size as set.
                    0: used for training, it's prohibited here.
            num_decoding_left_chunks (int, optional):
                    number of left chunks for decoding. Defaults to -1.
            simulate_streaming (bool, optional): simulate streaming inference. Defaults to False.
            reverse_weight (float, optional): reverse decoder weight, used by `attention_rescoring`.

        Raises:
            ValueError: when not support decoding_method.

        Returns:
            List[List[int]]: transcripts.
        """
        batch_size = feats.shape[0]
        if decoding_method in ['ctc_prefix_beam_search',
                               'attention_rescoring'] and batch_size > 1:
            logger.error(
                f'decoding mode {decoding_method} must be running with batch_size == 1'
            )
            logger.error(f"current batch_size is {batch_size}")
            sys.exit(1)
        if decoding_method == 'attention':
            hyps = self.recognize(
                feats,
                feats_lengths,
                beam_size=beam_size,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks,
                simulate_streaming=simulate_streaming)
            hyps = [hyp.tolist() for hyp in hyps]
        elif decoding_method == 'ctc_greedy_search':
            hyps = self.ctc_greedy_search(
                feats,
                feats_lengths,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks,
                simulate_streaming=simulate_streaming)
        # ctc_prefix_beam_search and attention_rescoring only return one
        # result in List[int], change it to List[List[int]] for compatible
        # with other batch decoding mode
        elif decoding_method == 'ctc_prefix_beam_search':
            assert feats.shape[0] == 1
            hyp = self.ctc_prefix_beam_search(
                feats,
                feats_lengths,
                beam_size,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks,
                simulate_streaming=simulate_streaming)
            hyps = [hyp]
        elif decoding_method == 'attention_rescoring':
            assert feats.shape[0] == 1
            hyp = self.attention_rescoring(
                feats,
                feats_lengths,
                beam_size,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks,
                ctc_weight=ctc_weight,
                simulate_streaming=simulate_streaming,
                reverse_weight=reverse_weight)
            hyps = [hyp]
        else:
            raise ValueError(f"Not support decoding method: {decoding_method}")

        res = [text_feature.defeaturize(hyp) for hyp in hyps]
        res_tokenids = [hyp for hyp in hyps]
        return res, res_tokenids


class U2DecodeModel(U2BaseModel):
    def scorers(self):
        """Scorers."""
        return dict(
            decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, x):
        """Encode acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: paddle.Tensor
        """
        self.eval()
        x = paddle.to_tensor(x).unsqueeze(0)
        ilen = paddle.shape(x)[1]
        enc_output, _ = self._forward_encoder(x, ilen)
        return enc_output.squeeze(0)


class U2Model(U2DecodeModel):
    def __init__(self, configs: dict):
        model_conf = configs.get('model_conf', dict())
        init_type = model_conf.get("init_type", None)
        with DefaultInitializerContext(init_type):
            vocab_size, encoder, decoder, ctc = U2Model._init_from_config(
                configs)
        super().__init__(
            vocab_size=vocab_size,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            **model_conf)

    @classmethod
    def _init_from_config(cls, configs: dict):
        """init sub module for model.

        Args:
            configs (dict): config dict.

        Raises:
            ValueError: raise when using not support encoder type.

        Returns:
            int, nn.Layer, nn.Layer, nn.Layer: vocab size, encoder, decoder, ctc
        """
        # cmvn
        if 'cmvn_file' in configs and configs['cmvn_file']:
            mean, istd = load_cmvn(configs['cmvn_file'],
                                   configs['cmvn_file_type'])
            global_cmvn = GlobalCMVN(
                paddle.to_tensor(mean, dtype=paddle.float),
                paddle.to_tensor(istd, dtype=paddle.float))
        else:
            global_cmvn = None

        # input & output dim
        input_dim = configs['input_dim']
        vocab_size = configs['output_dim']
        assert input_dim != 0, input_dim
        assert vocab_size != 0, vocab_size

        # encoder
        encoder_type = configs.get('encoder', 'transformer')
        logger.debug(f"U2 Encoder type: {encoder_type}")
        if encoder_type == 'transformer':
            encoder = TransformerEncoder(
                input_dim, global_cmvn=global_cmvn, **configs['encoder_conf'])
        elif encoder_type == 'conformer':
            encoder = ConformerEncoder(
                input_dim, global_cmvn=global_cmvn, **configs['encoder_conf'])
        elif encoder_type == 'squeezeformer':
            encoder = SqueezeformerEncoder(
                input_dim, global_cmvn=global_cmvn, **configs['encoder_conf'])
        else:
            raise ValueError(f"not support encoder type:{encoder_type}")

        # decoder
        decoder_type = configs.get('decoder', 'transformer')
        logger.debug(f"U2 Decoder type: {decoder_type}")
        if decoder_type == 'transformer':
            configs['model_conf'].pop('reverse_weight', None)
            configs['decoder_conf'].pop('r_num_blocks', None)
            decoder = TransformerDecoder(vocab_size,
                                         encoder.output_size(),
                                         **configs['decoder_conf'])
        elif decoder_type == 'bitransformer':
            assert 0.0 < configs['model_conf']['reverse_weight'] < 1.0
            assert configs['decoder_conf']['r_num_blocks'] > 0
            decoder = BiTransformerDecoder(vocab_size,
                                           encoder.output_size(),
                                           **configs['decoder_conf'])
        else:
            raise ValueError(f"not support decoder type:{decoder_type}")
        # ctc decoder and ctc loss
        model_conf = configs.get('model_conf', dict())
        dropout_rate = model_conf.get('ctc_dropout_rate', 0.0)
        grad_norm_type = model_conf.get('ctc_grad_norm_type', None)
        ctc = CTCDecoderBase(
            odim=vocab_size,
            enc_n_units=encoder.output_size(),
            blank_id=0,
            dropout_rate=dropout_rate,
            reduction=True,  # sum
            batch_average=True,  # sum / batch_size
            grad_norm_type=grad_norm_type)

        return vocab_size, encoder, decoder, ctc

    @classmethod
    def from_config(cls, configs: dict):
        """init model.

        Args:
            configs (dict): config dict.

        Raises:
            ValueError: raise when using not support encoder type.

        Returns:
            nn.Layer: U2Model
        """
        model = cls(configs)
        return model

    @classmethod
    def from_pretrained(cls, dataloader, config, checkpoint_path):
        """Build a DeepSpeech2Model model from a pretrained model.

        Args:
            dataloader (paddle.io.DataLoader): not used.
            config (yacs.config.CfgNode):  model configs
            checkpoint_path (Path or str): the path of pretrained model checkpoint, without extension name

        Returns:
            DeepSpeech2Model: The model built from pretrained result.
        """
        with UpdateConfig(config):
            config.input_dim = dataloader.feat_dim
            config.output_dim = dataloader.vocab_size

        model = cls.from_config(config)

        if checkpoint_path:
            infos = checkpoint.Checkpoint().load_parameters(
                model, checkpoint_path=checkpoint_path)
            logger.debug(f"checkpoint info: {infos}")
        layer_tools.summary(model)
        return model


class U2InferModel(U2Model):
    def __init__(self, configs: dict):
        super().__init__(configs)

        from paddlespeech.s2t.modules.fbank import KaldiFbank
        import yaml
        import json
        import numpy as np

        input_dim = configs['input_dim']
        process = configs['preprocess_config']
        with open(process, encoding="utf-8") as f:
            conf = yaml.safe_load(f)
            assert isinstance(conf, dict), type(self.conf)

        for idx, process in enumerate(conf['process']):
            assert isinstance(process, dict), type(process)
            opts = dict(process)
            process_type = opts.pop("type")

            if process_type == 'fbank_kaldi':
                opts.update({'n_mels': input_dim})
                opts['dither'] = 0.0
                self.fbank = KaldiFbank(**opts)
                logger.info(f"{self.__class__.__name__} export: {self.fbank}")
            if process_type == 'cmvn_json':
                # align with paddlespeech.audio.transform.cmvn:GlobalCMVN
                std_floor = 1.0e-20

                cmvn = opts['cmvn_path']
                if isinstance(cmvn, dict):
                    cmvn_stats = cmvn
                else:
                    with open(cmvn) as f:
                        cmvn_stats = json.load(f)
                count = cmvn_stats['frame_num']
                mean = np.array(cmvn_stats['mean_stat']) / count
                square_sums = np.array(cmvn_stats['var_stat'])
                var = square_sums / count - mean**2
                std = np.maximum(np.sqrt(var), std_floor)
                istd = 1.0 / std
                self.global_cmvn = GlobalCMVN(
                    paddle.to_tensor(mean, dtype=paddle.float),
                    paddle.to_tensor(istd, dtype=paddle.float))
                logger.info(
                    f"{self.__class__.__name__} export: {self.global_cmvn}")

    def forward(self,
                feats,
                feats_lengths,
                decoding_chunk_size=-1,
                num_decoding_left_chunks=-1,
                simulate_streaming=False):
        """export model function

        Args:
            feats (Tensor): [B, T, D]
            feats_lengths (Tensor): [B]

        Returns:
            List[List[int]]: best path result
        """
        # dummy code for dy2st
        # return self.ctc_greedy_search(
        #     feats,
        #     feats_lengths,
        #     decoding_chunk_size=decoding_chunk_size,
        #     num_decoding_left_chunks=num_decoding_left_chunks,
        #     simulate_streaming=simulate_streaming)
        return feats, feats_lengths

    def forward_feature(self, x):
        """feature pipeline.

        Args:
            x (paddle.Tensor): waveform (T,).

        Return:
            feat (paddle.Tensor): feature (T, D) 
        """
        x = paddle.cast(x, paddle.float32)
        feat = self.fbank(x)
        feat = self.global_cmvn(feat)
        return feat
