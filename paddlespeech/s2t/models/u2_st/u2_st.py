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
import time
from typing import Dict
from typing import Optional
from typing import Tuple

import paddle
from paddle import jit
from paddle import nn

from paddlespeech.audio.utils.tensor_utils import add_sos_eos
from paddlespeech.audio.utils.tensor_utils import th_accuracy
from paddlespeech.s2t.frontend.utility import IGNORE_ID
from paddlespeech.s2t.frontend.utility import load_cmvn
from paddlespeech.s2t.modules.cmvn import GlobalCMVN
from paddlespeech.s2t.modules.ctc import CTCDecoderBase
from paddlespeech.s2t.modules.decoder import TransformerDecoder
from paddlespeech.s2t.modules.encoder import ConformerEncoder
from paddlespeech.s2t.modules.encoder import TransformerEncoder
from paddlespeech.s2t.modules.loss import LabelSmoothingLoss
from paddlespeech.s2t.modules.mask import subsequent_mask
from paddlespeech.s2t.utils import checkpoint
from paddlespeech.s2t.utils import layer_tools
from paddlespeech.s2t.utils.log import Log
from paddlespeech.s2t.utils.utility import UpdateConfig

__all__ = ["U2STModel", "U2STInferModel"]

logger = Log(__name__).getlog()


class U2STBaseModel(nn.Layer):
    """CTC-Attention hybrid Encoder-Decoder model"""

    def __init__(self,
                 vocab_size: int,
                 encoder: TransformerEncoder,
                 st_decoder: TransformerDecoder,
                 decoder: TransformerDecoder=None,
                 ctc: CTCDecoderBase=None,
                 ctc_weight: float=0.0,
                 asr_weight: float=0.0,
                 ignore_id: int=IGNORE_ID,
                 lsm_weight: float=0.0,
                 length_normalized_loss: bool=False,
                 **kwargs):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.asr_weight = asr_weight

        self.encoder = encoder
        self.st_decoder = st_decoder
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
            asr_text: paddle.Tensor=None,
            asr_text_lengths: paddle.Tensor=None,
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

        # 2a. ST-decoder branch
        start = time.time()
        loss_st, acc_st = self._calc_st_loss(encoder_out, encoder_mask, text,
                                             text_lengths)
        decoder_time = time.time() - start

        loss_asr_att = None
        loss_asr_ctc = None
        # 2b. ASR Attention-decoder branch
        if self.asr_weight > 0.:
            if self.ctc_weight != 1.0:
                start = time.time()
                loss_asr_att, acc_att = self._calc_att_loss(
                    encoder_out, encoder_mask, asr_text, asr_text_lengths)
                decoder_time = time.time() - start

            # 2c. CTC branch
            if self.ctc_weight != 0.0:
                start = time.time()
                loss_asr_ctc = self.ctc(encoder_out, encoder_out_lens, asr_text,
                                        asr_text_lengths)
                ctc_time = time.time() - start

            if loss_asr_ctc is None:
                loss_asr = loss_asr_att
            elif loss_asr_att is None:
                loss_asr = loss_asr_ctc
            else:
                loss_asr = self.ctc_weight * loss_asr_ctc + (1 - self.ctc_weight
                                                             ) * loss_asr_att
            loss = self.asr_weight * loss_asr + (1 - self.asr_weight) * loss_st
        else:
            loss = loss_st
        return loss, loss_st, loss_asr_att, loss_asr_ctc

    def _calc_st_loss(
            self,
            encoder_out: paddle.Tensor,
            encoder_mask: paddle.Tensor,
            ys_pad: paddle.Tensor,
            ys_pad_lens: paddle.Tensor, ) -> Tuple[paddle.Tensor, float]:
        """Calc attention loss.

        Args:
            encoder_out (paddle.Tensor): [B, Tmax, D]
            encoder_mask (paddle.Tensor): [B, 1, Tmax]
            ys_pad (paddle.Tensor): [B, Umax]
            ys_pad_lens (paddle.Tensor): [B]

        Returns:
            Tuple[paddle.Tensor, float]: attention_loss, accuracy rate
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos,
                                            self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, *_ = self.st_decoder(encoder_out, encoder_mask, ys_in_pad,
                                          ys_in_lens)

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.reshape([-1, self.vocab_size]),
            ys_out_pad,
            ignore_label=self.ignore_id, )
        return loss_att, acc_att

    def _calc_att_loss(
            self,
            encoder_out: paddle.Tensor,
            encoder_mask: paddle.Tensor,
            ys_pad: paddle.Tensor,
            ys_pad_lens: paddle.Tensor, ) -> Tuple[paddle.Tensor, float]:
        """Calc attention loss.

        Args:
            encoder_out (paddle.Tensor): [B, Tmax, D]
            encoder_mask (paddle.Tensor): [B, 1, Tmax]
            ys_pad (paddle.Tensor): [B, Umax]
            ys_pad_lens (paddle.Tensor): [B]

        Returns:
            Tuple[paddle.Tensor, float]: attention_loss, accuracy rate
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos,
                                            self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, *_ = self.decoder(encoder_out, encoder_mask, ys_in_pad,
                                       ys_in_lens)

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
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

    def translate(
            self,
            speech: paddle.Tensor,
            speech_lengths: paddle.Tensor,
            beam_size: int=10,
            word_reward: float=0.0,
            maxlenratio: float=0.5,
            decoding_chunk_size: int=-1,
            num_decoding_left_chunks: int=-1,
            simulate_streaming: bool=False, ) -> paddle.Tensor:
        """ Apply beam search on attention decoder with length penalty
        Args:
            speech (paddle.Tensor): (batch, max_len, feat_dim)
            speech_length (paddle.Tensor): (batch, )
            beam_size (int): beam size for beam search
            word_reward (float): word reward used in beam search
            maxlenratio (float): max length ratio to bound the length of translated text
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
        assert speech.shape[0] == 1
        device = speech.place

        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder and init hypothesis
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)

        maxlen = max(int(encoder_out.shape[1] * maxlenratio), 5)

        hyp = {"score": 0.0, "yseq": [self.sos], "cache": None}
        hyps = [hyp]
        ended_hyps = []
        cur_best_score = -float("inf")
        cache = None

        # 2. Decoder forward step by step
        for i in range(1, maxlen + 1):
            ys = paddle.ones((len(hyps), i), dtype=paddle.long)

            if hyps[0]["cache"] is not None:
                cache = [
                    paddle.ones(
                        (len(hyps), i - 1, hyp_cache.shape[-1]),
                        dtype=paddle.float32) for hyp_cache in hyps[0]["cache"]
                ]
            for j, hyp in enumerate(hyps):
                ys[j, :] = paddle.to_tensor(hyp["yseq"])
                if hyps[0]["cache"] is not None:
                    for k in range(len(cache)):
                        cache[k][j] = hyps[j]["cache"][k]
            ys_mask = subsequent_mask(i).unsqueeze(0).to(device)

            logp, cache = self.st_decoder.forward_one_step(
                encoder_out.repeat(len(hyps), 1, 1),
                encoder_mask.repeat(len(hyps), 1, 1), ys, ys_mask, cache)

            hyps_best_kept = []
            for j, hyp in enumerate(hyps):
                top_k_logp, top_k_index = logp[j:j + 1].topk(beam_size)

                for b in range(beam_size):
                    new_hyp = {}
                    new_hyp["score"] = hyp["score"] + float(top_k_logp[0, b])
                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                    new_hyp["yseq"][:len(hyp["yseq"])] = hyp["yseq"]
                    new_hyp["yseq"][len(hyp["yseq"])] = int(top_k_index[0, b])
                    new_hyp["cache"] = [cache_[j] for cache_ in cache]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: -x["score"])[:beam_size]

            # sort and get nbest
            hyps = hyps_best_kept
            if i == maxlen:
                for hyp in hyps:
                    hyp["yseq"].append(self.eos)

            # finalize the ended hypotheses with word reward (by length)
            remained_hyps = []
            for hyp in hyps:
                if hyp["yseq"][-1] == self.eos:
                    hyp["score"] += (i - 1) * word_reward
                    cur_best_score = max(cur_best_score, hyp["score"])
                    ended_hyps.append(hyp)
                else:
                    # stop while guarantee the optimality
                    if hyp["score"] + maxlen * word_reward > cur_best_score:
                        remained_hyps.append(hyp)

            # stop predition when there is no unended hypothesis
            if not remained_hyps:
                break
            hyps = remained_hyps

        # 3. Select best of best
        best_hyp = max(ended_hyps, key=lambda x: x["score"])

        return paddle.to_tensor([best_hyp["yseq"][1:]])

    # @jit.to_static
    def subsampling_rate(self) -> int:
        """ Export interface for c++ call, return subsampling_rate of the
            model
        """
        return self.encoder.embed.subsampling_rate

    # @jit.to_static
    def right_context(self) -> int:
        """ Export interface for c++ call, return right_context of the model
        """
        return self.encoder.embed.right_context

    # @jit.to_static
    def sos_symbol(self) -> int:
        """ Export interface for c++ call, return sos symbol id of the model
        """
        return self.sos

    # @jit.to_static
    def eos_symbol(self) -> int:
        """ Export interface for c++ call, return eos symbol id of the model
        """
        return self.eos

    @jit.to_static
    def forward_encoder_chunk(
            self,
            xs: paddle.Tensor,
            offset: int,
            required_cache_size: int,
            att_cache: paddle.Tensor=paddle.zeros([0, 0, 0, 0]),
            cnn_cache: paddle.Tensor=paddle.zeros([0, 0, 0, 0]),
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
                `cache_t2 == cnn.lorder - 1`

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
            xs (paddle.Tensor): encoder output
        Returns:
            paddle.Tensor: activation before ctc
        """
        return self.ctc.log_softmax(xs)

    @jit.to_static
    def forward_attention_decoder(
            self,
            hyps: paddle.Tensor,
            hyps_lens: paddle.Tensor,
            encoder_out: paddle.Tensor, ) -> paddle.Tensor:
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
        # (num_hyps, max_hyps_len, vocab_size)
        decoder_out, _ = self.decoder(encoder_out, encoder_mask, hyps,
                                      hyps_lens)
        decoder_out = paddle.nn.functional.log_softmax(decoder_out, dim=-1)
        return decoder_out

    @paddle.no_grad()
    def decode(self,
               feats: paddle.Tensor,
               feats_lengths: paddle.Tensor,
               text_feature: Dict[str, int],
               decoding_method: str,
               beam_size: int,
               word_reward: float=0.0,
               maxlenratio: float=0.5,
               decoding_chunk_size: int=-1,
               num_decoding_left_chunks: int=-1,
               simulate_streaming: bool=False):
        """u2 decoding.

        Args:
            feats (Tensor): audio features, (B, T, D)
            feats_lengths (Tensor): (B)
            text_feature (TextFeaturizer): text feature object.
            decoding_method (str): decoding mode, e.g.
                    'fullsentence',
                    'simultaneous'
            beam_size (int): beam size for search
            decoding_chunk_size (int, optional): decoding chunk size. Defaults to -1.
                    <0: for decoding, use full chunk.
                    >0: for decoding, use fixed chunk size as set.
                    0: used for training, it's prohibited here.
            num_decoding_left_chunks (int, optional):
                    number of left chunks for decoding. Defaults to -1.
            simulate_streaming (bool, optional): simulate streaming inference. Defaults to False.

        Raises:
            ValueError: when not support decoding_method.

        Returns:
            List[List[int]]: transcripts.
        """
        batch_size = feats.shape[0]

        if decoding_method == 'fullsentence':
            hyps = self.translate(
                feats,
                feats_lengths,
                beam_size=beam_size,
                word_reward=word_reward,
                maxlenratio=maxlenratio,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks,
                simulate_streaming=simulate_streaming)
            hyps = [hyp.tolist() for hyp in hyps]
        else:
            raise ValueError(f"Not support decoding method: {decoding_method}")

        res = [text_feature.defeaturize(hyp) for hyp in hyps]
        return res


class U2STModel(U2STBaseModel):
    def __init__(self, configs: dict):
        vocab_size, encoder, decoder = U2STModel._init_from_config(configs)

        if isinstance(decoder, Tuple):
            st_decoder, asr_decoder, ctc = decoder
            super().__init__(
                vocab_size=vocab_size,
                encoder=encoder,
                st_decoder=st_decoder,
                decoder=asr_decoder,
                ctc=ctc,
                **configs['model_conf'])
        else:
            super().__init__(
                vocab_size=vocab_size,
                encoder=encoder,
                st_decoder=decoder,
                **configs['model_conf'])

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
        if configs['cmvn_file'] is not None:
            mean, istd = load_cmvn(configs['cmvn_file'],
                                   configs['cmvn_file_type'])
            global_cmvn = GlobalCMVN(
                paddle.to_tensor(mean, dtype=paddle.float),
                paddle.to_tensor(istd, dtype=paddle.float))
        else:
            global_cmvn = None

        input_dim = configs['input_dim']
        vocab_size = configs['output_dim']
        assert input_dim != 0, input_dim
        assert vocab_size != 0, vocab_size

        encoder_type = configs.get('encoder', 'transformer')
        logger.info(f"U2 Encoder type: {encoder_type}")
        if encoder_type == 'transformer':
            encoder = TransformerEncoder(
                input_dim, global_cmvn=global_cmvn, **configs['encoder_conf'])
        elif encoder_type == 'conformer':
            encoder = ConformerEncoder(
                input_dim, global_cmvn=global_cmvn, **configs['encoder_conf'])
        else:
            raise ValueError(f"not support encoder type:{encoder_type}")

        st_decoder = TransformerDecoder(vocab_size,
                                        encoder.output_size(),
                                        **configs['decoder_conf'])

        asr_weight = configs['model_conf']['asr_weight']
        logger.info(f"ASR Joint Training Weight: {asr_weight}")

        if asr_weight > 0.:
            decoder = TransformerDecoder(vocab_size,
                                         encoder.output_size(),
                                         **configs['decoder_conf'])
            # ctc decoder and ctc loss
            model_conf = configs['model_conf']
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

            return vocab_size, encoder, (st_decoder, decoder, ctc)
        else:
            return vocab_size, encoder, st_decoder

    @classmethod
    def from_config(cls, configs: dict):
        """init model.

        Args:
            configs (dict): config dict.

        Raises:
            ValueError: raise when using not support encoder type.

        Returns:
            nn.Layer: U2STModel
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
            config.input_dim = dataloader.collate_fn.feature_size
            config.output_dim = dataloader.collate_fn.vocab_size

        model = cls.from_config(config)

        if checkpoint_path:
            infos = checkpoint.load_parameters(
                model, checkpoint_path=checkpoint_path)
            logger.info(f"checkpoint info: {infos}")
        layer_tools.summary(model)
        return model


class U2STInferModel(U2STModel):
    def __init__(self, configs: dict):
        super().__init__(configs)

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
        return self.translate(
            feats,
            feats_lengths,
            decoding_chunk_size=decoding_chunk_size,
            num_decoding_left_chunks=num_decoding_left_chunks,
            simulate_streaming=simulate_streaming)
