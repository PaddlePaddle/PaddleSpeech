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
from typing import Union

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
from paddlespeech.s2t.models.transducer.greedy_search import basic_greedy_search
from paddlespeech.s2t.models.transducer.joint import TransducerJoint
from paddlespeech.s2t.models.transducer.predictor import ConvPredictor
from paddlespeech.s2t.models.transducer.predictor import EmbeddingPredictor
from paddlespeech.s2t.models.transducer.predictor import PredictorBase
from paddlespeech.s2t.models.transducer.predictor import RNNPredictor
from paddlespeech.s2t.models.transducer.prefix_beam_search import PrefixBeamSearch
from paddlespeech.s2t.modules.cmvn import GlobalCMVN
from paddlespeech.s2t.modules.ctc import CTCDecoderBase
from paddlespeech.s2t.modules.decoder import BiTransformerDecoder
from paddlespeech.s2t.modules.decoder import TransformerDecoder
from paddlespeech.s2t.modules.encoder import ConformerEncoder
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
from paddlespeech.s2t.utils.rnnt_utils import add_blank
from paddlespeech.s2t.utils.utility import log_add
from paddlespeech.s2t.utils.utility import UpdateConfig

__all__ = ["TransducerBaseModel", "TransducerModel"]

logger = Log(__name__).getlog()


class TransducerBaseModel(ASRInterface, nn.Layer):
    """Transducer-ctc-attention hybrid Encoder-Predictor-Decoder model"""

    def __init__(
            self,
            vocab_size: int,
            blank: int,
            encoder: nn.Layer,
            predictor: PredictorBase,
            joint: nn.Layer,
            attention_decoder: Optional[Union[TransformerDecoder,
                                              BiTransformerDecoder]]=None,
            ctc: Optional[CTCDecoderBase]=None,
            ctc_weight: float=0,
            ignore_id: int=IGNORE_ID,
            reverse_weight: float=0.0,
            lsm_weight: float=0.0,
            length_normalized_loss: bool=False,
            transducer_weight: float=1.0,
            attention_weight: float=0.0,
            **kwargs, ):
        assert attention_weight + ctc_weight + transducer_weight == 1.0

        nn.Layer.__init__(self)

        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.blank = blank
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight
        self.transducer_weight = transducer_weight
        self.attention_decoder_weight = 1 - self.transducer_weight - self.ctc_weight

        self.encoder = encoder
        self.attention_decoder = attention_decoder
        self.ctc = ctc
        self.predictor = predictor
        self.joint = joint
        self.bs = None

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
    ) -> Dict[str, Optional[paddle.Tensor]]:
        """Frontend + Encoder + predictor + joint + loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)

        # Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)  #[B, 1, T] -> [B]
        # predictor
        ys_in_pad = add_blank(text, self.blank, self.ignore_id)
        predictor_out = self.predictor(ys_in_pad)
        # joint
        joint_out = self.joint(encoder_out, predictor_out)
        # NOTE(Mddct): some loss implementation require pad valid is zero
        # torch.int32 rnnt_loss required
        rnnt_text = text
        rnnt_text = paddle.where(rnnt_text == self.ignore_id, 0, rnnt_text)
        rnnt_text = rnnt_text.astype(paddle.int32)
        rnnt_text_lengths = text_lengths.astype(paddle.int32)
        rnnt_encoder_out_lens = encoder_out_lens.astype(paddle.int32)
        loss = paddle.nn.functional.rnnt_loss(
            joint_out,
            rnnt_text,
            rnnt_encoder_out_lens,
            rnnt_text_lengths,
            blank=self.blank,
            reduction="mean")
        loss_rnnt = loss

        loss = self.transducer_weight * loss
        # optional attention decoder
        loss_att: Optional[paddle.Tensor] = None
        if self.attention_decoder_weight != 0.0 and self.attention_decoder is not None:
            loss_att, _ = self._calc_att_loss(encoder_out, encoder_mask, text,
                                              text_lengths, self.reverse_weight)

        # optional ctc
        loss_ctc: Optional[paddle.Tensor] = None
        if self.ctc_weight != 0.0 and self.ctc is not None:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, text,
                                text_lengths)
        else:
            loss_ctc = None

        if loss_ctc is not None:
            loss = loss + self.ctc_weight * paddle.sum(loss_ctc)
        if loss_att is not None:
            loss = loss + self.attention_decoder_weight * paddle.sum(loss_att)
        # NOTE: 'loss' must be in dict

        return {
            'loss': loss,
            'loss_att': loss_att,
            'loss_ctc': loss_ctc,
            'loss_rnnt': loss_rnnt,
        }

    def init_bs(self):
        if self.bs is None:
            self.bs = PrefixBeamSearch(self.encoder, self.predictor, self.joint,
                                       self.ctc, self.blank)

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
        decoder_out, r_decoder_out, _ = self.attention_decoder(
            encoder_out, encoder_mask, ys_in_pad, ys_in_lens, r_ys_in_pad,
            reverse_weight)

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        r_loss_att = paddle.to_tensor(0.0)
        if reverse_weight > 0.0:
            r_loss_att = self.criterion_att(r_decoder_out, r_ys_out_pad)
        loss_att = loss_att * (1 - reverse_weight) + r_loss_att * reverse_weight
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id, )
        return loss_att, acc_att

    def _cal_transducer_score(
            self,
            encoder_out: paddle.Tensor,
            encoder_mask: paddle.Tensor,
            hyps_lens: paddle.Tensor,
            hyps_pad: paddle.Tensor, ):
        # ignore id -> blank, add blank at head
        hyps_pad_blank = add_blank(hyps_pad, self.blank, self.ignore_id)
        xs_in_lens = encoder_mask.squeeze(1).sum(1).int()

        # 1. Forward predictor
        predictor_out = self.predictor(hyps_pad_blank)
        # 2. Forward joint
        joint_out = self.joint(encoder_out, predictor_out)
        rnnt_text = hyps_pad.astype(paddle.int64)
        rnnt_text = paddle.where(rnnt_text == self.ignore_id, 0,
                                 rnnt_text).astype(paddle.int32)
        # 3. Compute transducer loss
        loss_td = paddle.nn.functional.rnnt_loss(
            joint_out,
            rnnt_text,
            xs_in_lens,
            hyps_lens.int(),
            blank=self.blank,
            reduction='none')
        return loss_td * -1

    def _cal_attn_score(
            self,
            encoder_out: paddle.Tensor,
            encoder_mask: paddle.Tensor,
            hyps_pad: paddle.Tensor,
            hyps_lens: paddle.Tensor, ):
        # (beam_size, max_hyps_len)
        ori_hyps_pad = hyps_pad

        # td_score = loss_td * -1
        hyps_pad, _ = add_sos_eos(hyps_pad, self.sos, self.eos, self.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining
        # used for right to left decoder
        r_hyps_pad = reverse_pad_list(ori_hyps_pad, hyps_lens, self.ignore_id)
        r_hyps_pad, _ = add_sos_eos(r_hyps_pad, self.sos, self.eos,
                                    self.ignore_id)
        decoder_out, r_decoder_out, _ = self.attention_decoder(
            encoder_out, encoder_mask, hyps_pad, hyps_lens, r_hyps_pad,
            self.reverse_weight)  # (beam_size, max_hyps_len, vocab_size)
        decoder_out = paddle.nn.functional.log_softmax(decoder_out, axis=-1)
        decoder_out = decoder_out.cpu().numpy()
        # r_decoder_out will be 0.0, if reverse_weight is 0.0 or decoder is a
        # conventional transformer decoder.
        r_decoder_out = paddle.nn.functional.log_softmax(r_decoder_out, axis=-1)
        r_decoder_out = r_decoder_out.cpu().numpy()
        return decoder_out, r_decoder_out

    def beam_search(
            self,
            speech: paddle.Tensor,
            speech_lengths: paddle.Tensor,
            decoding_chunk_size: int=-1,
            beam_size: int=5,
            num_decoding_left_chunks: int=-1,
            simulate_streaming: bool=False,
            ctc_weight: float=0.3,
            transducer_weight: float=0.7, ):
        """beam search

        Args:
            speech (torch.Tensor): (batch=1, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
            ctc_weight (float): ctc probability weight in transducer
                prefix beam search.
                final_prob = ctc_weight * ctc_prob + transducer_weight * transducer_prob
            transducer_weight (float): transducer probability weight in
                prefix beam search
        Returns:
            List[List[int]]: best path result

        """
        self.init_bs()
        beam, _ = self.bs.prefix_beam_search(
            speech,
            speech_lengths,
            decoding_chunk_size,
            beam_size,
            num_decoding_left_chunks,
            simulate_streaming,
            ctc_weight,
            transducer_weight, )
        return beam[0].hyp[1:], beam[0].score

    def transducer_attention_rescoring(
            self,
            speech: paddle.Tensor,
            speech_lengths: paddle.Tensor,
            beam_size: int,
            decoding_chunk_size: int=-1,
            num_decoding_left_chunks: int=-1,
            simulate_streaming: bool=False,
            reverse_weight: float=0.0,
            ctc_weight: float=0.0,
            attn_weight: float=0.0,
            transducer_weight: float=0.0,
            search_ctc_weight: float=1.0,
            search_transducer_weight: float=0.0,
            beam_search_type: str='transducer') -> List[List[int]]:
        """beam search

        Args:
            speech (torch.Tensor): (batch=1, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
            ctc_weight (float): ctc probability weight using in rescoring.
                rescore_prob = ctc_weight * ctc_prob +
                               transducer_weight * (transducer_loss * -1) +
                               attn_weight * attn_prob
            attn_weight (float): attn probability weight using in rescoring.
            transducer_weight (float): transducer probability weight using in
                rescoring
            search_ctc_weight (float): ctc weight using
                               in rnnt beam search (seeing in self.beam_search)
            search_transducer_weight (float): transducer weight using
                               in rnnt beam search (seeing in self.beam_search)
        Returns:
            List[List[int]]: best path result

        """

        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        if reverse_weight > 0.0:
            # decoder should be a bitransformer decoder if reverse_weight > 0.0
            assert hasattr(self.attention_decoder, 'right_decoder')
        device = speech.device
        batch_size = speech.shape[0]
        # For attention rescoring we only support batch_size=1
        assert batch_size == 1
        # encoder_out: (1, maxlen, encoder_dim), len(hyps) = beam_size
        self.init_bs()
        if beam_search_type == 'transducer':
            beam, encoder_out = self.bs.prefix_beam_search(
                speech,
                speech_lengths,
                decoding_chunk_size=decoding_chunk_size,
                beam_size=beam_size,
                num_decoding_left_chunks=num_decoding_left_chunks,
                ctc_weight=search_ctc_weight,
                transducer_weight=search_transducer_weight, )
            beam_score = [s.score for s in beam]
            hyps = [s.hyp[1:] for s in beam]

        elif beam_search_type == 'ctc':
            hyps, encoder_out = self._ctc_prefix_beam_search(
                speech,
                speech_lengths,
                beam_size=beam_size,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks,
                simulate_streaming=simulate_streaming)
            beam_score = [hyp[1] for hyp in hyps]
            hyps = [hyp[0] for hyp in hyps]
        assert len(hyps) == beam_size

        # build hyps and encoder output
        hyps_pad = pad_sequence([
            paddle.to_tensor(hyp, device=device, dtype=paddle.long)
            for hyp in hyps
        ], True, self.ignore_id)  # (beam_size, max_hyps_len)
        hyps_lens = paddle.to_tensor(
            [len(hyp) for hyp in hyps], device=device,
            dtype=paddle.long)  # (beam_size,)

        encoder_out = encoder_out.repeat(beam_size, 1, 1)
        encoder_mask = paddle.ones(
            beam_size, 1, encoder_out.size(1), dtype=paddle.bool, device=device)

        # 2.1 calculate transducer score
        td_score = self._cal_transducer_score(
            encoder_out,
            encoder_mask,
            hyps_lens,
            hyps_pad, )
        # 2.2 calculate attention score
        decoder_out, r_decoder_out = self._cal_attn_score(
            encoder_out,
            encoder_mask,
            hyps_pad,
            hyps_lens, )

        # Only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0
        for i, hyp in enumerate(hyps):
            score = 0.0
            for j, w in enumerate(hyp):
                score += decoder_out[i][j][w]
            score += decoder_out[i][len(hyp)][self.eos]
            td_s = td_score[i]
            # add right to left decoder score
            if reverse_weight > 0:
                r_score = 0.0
                for j, w in enumerate(hyp):
                    r_score += r_decoder_out[i][len(hyp) - j - 1][w]
                r_score += r_decoder_out[i][len(hyp)][self.eos]
                score = score * (1 - reverse_weight) + r_score * reverse_weight
            # add ctc score
            score = score * attn_weight + \
                beam_score[i] * ctc_weight + \
                td_s * transducer_weight
            if score > best_score:
                best_score = score
                best_index = i

        return hyps[best_index], best_score

    def greedy_search(
            self,
            speech: paddle.Tensor,
            speech_lengths: paddle.Tensor,
            decoding_chunk_size: int=-1,
            num_decoding_left_chunks: int=-1,
            simulate_streaming: bool=False,
            n_steps: int=64, ) -> List[List[int]]:
        """ greedy search

        Args:
            speech (paddle.Tensor): (batch=1, max_len, feat_dim)
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
        # TODO(Mddct): batch decode
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0

        # TODO(Mddct): forward chunk by chunk
        _ = simulate_streaming
        # Let's assume B = batch_size

        encoder_out, encoder_mask = self.encoder(
            speech,
            speech_lengths,
            decoding_chunk_size,
            num_decoding_left_chunks, )
        print("111", speech.shape, decoding_chunk_size)
        encoder_out_lens = encoder_mask.squeeze(1).sum()
        hyps = basic_greedy_search(
            self, encoder_out, encoder_out_lens, n_steps=n_steps)

        return hyps

    def forward_encoder_chunk(
            self,
            xs: paddle.Tensor,
            offset: int,
            required_cache_size: int,
            att_cache: paddle.Tensor=paddle.zeros([0, 0, 0, 0]),
            cnn_cache: paddle.Tensor=paddle.zeros([0, 0, 0, 0]),
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:

        return self.encoder.forward_chunk(xs, offset, required_cache_size,
                                          att_cache, cnn_cache)

    def forward_predictor_step(self,
                               xs: paddle.Tensor,
                               cache: List[paddle.Tensor]
                               ) -> Tuple[paddle.Tensor, List[paddle.Tensor]]:
        assert len(cache) == 2
        # fake padding
        padding = paddle.zeros([1, 1])
        return self.predictor.forward_step(xs, padding, cache)

    def forward_joint_step(self,
                           enc_out: paddle.Tensor,
                           pred_out: paddle.Tensor) -> paddle.Tensor:
        return self.joint(enc_out, pred_out)

    def forward_predictor_init_state(self) -> List[paddle.Tensor]:
        return self.predictor.init_state(1, device=paddle.device("cpu"))

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
        encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(
            running_size, maxlen, encoder_dim)  # (B*N, maxlen, encoder_dim)
        encoder_mask = encoder_mask.unsqueeze(1).repeat(
            1, beam_size, 1, 1).view(running_size, 1,
                                     maxlen)  # (B*N, 1, max_len)

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
            logp, cache = self.attention_decoder.forward_one_step(
                encoder_out, encoder_mask, hyps, hyps_mask, cache)
            # 2.2 First beam prune: select topk best prob at current time
            top_k_logp, top_k_index = logp.topk(beam_size)  # (B*N, N)
            top_k_logp = mask_finished_scores(top_k_logp, end_flag)
            top_k_index = mask_finished_preds(top_k_index, end_flag, self.eos)

            # 2.3 Seconde beam prune: select topk score with history
            scores = scores + top_k_logp  # (B*N, N), broadcast add
            scores = scores.view(batch_size, beam_size * beam_size)  # (B, N*N)
            scores, offset_k_index = scores.topk(k=beam_size)  # (B, N)
            scores = scores.view(-1, 1)  # (B*N, 1)

            # 2.4. Compute base index in top_k_index,
            # regard top_k_index as (B*N*N),regard offset_k_index as (B*N),
            # then find offset_k_index in top_k_index
            base_k_index = paddle.arange(batch_size).view(-1, 1).repeat(
                1, beam_size)  # (B, N)
            base_k_index = base_k_index * beam_size * beam_size
            best_k_index = base_k_index.view(-1) + offset_k_index.view(
                -1)  # (B*N)

            # 2.5 Update best hyps
            best_k_pred = paddle.index_select(
                top_k_index.view(-1), index=best_k_index, axis=0)  # (B*N)
            best_hyps_index = best_k_index // beam_size
            last_best_k_hyps = paddle.index_select(
                hyps, index=best_hyps_index, axis=0)  # (B*N, i)
            hyps = paddle.cat(
                (last_best_k_hyps, best_k_pred.view(-1, 1)),
                dim=1)  # (B*N, i+1)

            # 2.6 Update end flag
            end_flag = paddle.equal(hyps[:, -1], self.eos).view(-1, 1)

        # 3. Select best of best
        scores = scores.view(batch_size, beam_size)
        # TODO: length normalization
        best_index = paddle.argmax(scores, axis=-1).long()  # (B)
        best_hyps_index = best_index + paddle.arange(
            batch_size, dtype=paddle.long) * beam_size
        best_hyps = paddle.index_select(hyps, index=best_hyps_index, axis=0)
        best_hyps = best_hyps[:, 1:]
        return best_hyps

    @paddle.no_grad()
    def decode(
            self,
            feats: paddle.Tensor,
            feats_lengths: paddle.Tensor,
            text_feature: Dict[str, int],
            decoding_method: str,
            beam_size: int,
            ctc_weight: float=0.0,
            attn_weight: float=0.3,
            transducer_weight: float=0.7,
            search_ctc_weight: float=1.0,
            search_transducer_weight: float=0.0,
            decoding_chunk_size: int=-1,
            num_decoding_left_chunks: int=-1,
            simulate_streaming: bool=False,
            reverse_weight: float=0.0,
            beam_search_type: str='transducer', ):
        """rnnt decoding.

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
            hyps = self.greedy_search(
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
            hyp, _ = self.beam_search(
                feats,
                feats_lengths,
                beam_size=beam_size,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks,
                simulate_streaming=simulate_streaming,
                ctc_weight=ctc_weight,
                transducer_weight=transducer_weight)
            hyps = [hyp]
        elif decoding_method == 'attention_rescoring':
            assert feats.shape[0] == 1
            hyp, _ = self.transducer_attention_rescoring(
                feats,
                feats_lengths,
                beam_size,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks,
                ctc_weight=ctc_weight,
                simulate_streaming=simulate_streaming,
                reverse_weight=reverse_weight,
                attn_weight=attn_weight,
                transducer_weight=transducer_weight,
                search_ctc_weight=search_ctc_weight,
                search_transducer_weight=search_transducer_weight,
                beam_search_type=beam_search_type)
            hyps = [hyp]
        else:
            raise ValueError(f"Not support decoding method: {decoding_method}")

        res = [text_feature.defeaturize(hyp) for hyp in hyps]
        res_tokenids = [hyp for hyp in hyps]
        return res, res_tokenids


class TransducerModel(TransducerBaseModel):
    def __init__(self, configs: dict):
        model_conf = configs.get('model_conf', dict())
        init_type = model_conf.get("init_type", None)
        with DefaultInitializerContext(init_type):
            vocab_size, encoder, attention_decoder, ctc, predictor, joint = TransducerModel._init_from_config(
                configs)
        super().__init__(
            vocab_size=vocab_size,
            blank=0,
            encoder=encoder,
            attention_decoder=attention_decoder,
            ctc=ctc,
            predictor=predictor,
            joint=joint,
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
        encoder_type = configs.get('encoder', 'conformer')
        logger.debug(f"Transducer Model Encoder type: {encoder_type}")
        if encoder_type == 'transformer':
            encoder = TransformerEncoder(
                input_dim, global_cmvn=global_cmvn, **configs['encoder_conf'])
        elif encoder_type == 'conformer':
            encoder = ConformerEncoder(
                input_dim, global_cmvn=global_cmvn, **configs['encoder_conf'])
        else:
            raise ValueError(f"not support encoder type:{encoder_type}")

        # decoder
        decoder_type = configs.get('decoder', 'bitransformer')
        logger.debug(f"Transducer Model Decoder type: {decoder_type}")
        if decoder_type == 'transformer':
            attention_decoder = TransformerDecoder(vocab_size,
                                                   encoder.output_size(),
                                                   **configs['decoder_conf'])
        elif decoder_type == 'bitransformer':
            assert 0.0 < configs['model_conf']['reverse_weight'] < 1.0
            assert configs['decoder_conf']['r_num_blocks'] > 0
            attention_decoder = BiTransformerDecoder(vocab_size,
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

        # Init joint CTC/Attention or Transducer model
        if 'predictor' in configs:
            predictor_type = configs.get('predictor', 'rnn')
            if predictor_type == 'rnn':
                predictor = RNNPredictor(vocab_size,
                                         **configs['predictor_conf'])
            elif predictor_type == 'embedding':
                predictor = EmbeddingPredictor(vocab_size,
                                               **configs['predictor_conf'])
                configs['predictor_conf']['output_size'] = configs[
                    'predictor_conf']['embed_size']
            elif predictor_type == 'conv':
                predictor = ConvPredictor(vocab_size,
                                          **configs['predictor_conf'])
                configs['predictor_conf']['output_size'] = configs[
                    'predictor_conf']['embed_size']
            else:
                raise NotImplementedError(
                    "only rnn, embedding and conv type support now")
            configs['joint_conf']['enc_output_size'] = configs['encoder_conf'][
                'output_size']
            configs['joint_conf']['pred_output_size'] = configs[
                'predictor_conf']['output_size']
            joint = TransducerJoint(vocab_size, **configs['joint_conf'])

        return vocab_size, encoder, attention_decoder, ctc, predictor, joint

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
