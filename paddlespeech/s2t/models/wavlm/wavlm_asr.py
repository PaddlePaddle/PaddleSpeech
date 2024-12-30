# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from collections import defaultdict
from typing import Dict
from typing import List
from typing import Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .wavlm_paddle import WavLM
from .wavlm_paddle import WavLMConfig
from paddlespeech.s2t.models.wav2vec2.modules.VanillaNN import VanillaNN
from paddlespeech.s2t.models.wav2vec2.processing.speech_augmentation import SpecAugment
from paddlespeech.s2t.modules.ctc import CTCDecoderBase as CTC
from paddlespeech.s2t.modules.initializer import DefaultInitializerContext
from paddlespeech.s2t.utils.ctc_utils import remove_duplicates_and_blank
from paddlespeech.s2t.utils.utility import log_add


class WavLMASR(nn.Layer):
    def __init__(self, config: dict):
        super().__init__()
        init_type = config.get("init_type", None)
        with DefaultInitializerContext(init_type):
            self.config = config
            wavlm_config = WavLMConfig(config)
            wavlm = WavLM(wavlm_config)

            self.normalize_wav = config.normalize_wav
            self.output_norm = config.output_norm
            if hasattr(config, 'spec_augment'):
                self.spec_augment = SpecAugment(**config.spec_augment)

            if config.freeze_wavlm:
                wavlm.eval()
                for parm in wavlm.parameters():
                    parm.trainable = False
            self.wavlm = wavlm
            self.enc = VanillaNN(**config.enc)
            self.ctc = CTC(**config.ctc,
                           odim=config.output_dim,
                           batch_average=False,
                           reduction='mean')

    def forward(self, wav, wavs_lens_rate, target, target_lens):
        if self.normalize_wav:
            wav = F.layer_norm(wav, wav.shape[1:])

        # Extract wav2vec output
        out = self.wavlm(wav)
        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out, out.shape[1:])

        if self.training and hasattr(self.config, 'spec_augment'):
            feats = self.spec_augment(out)
        else:
            feats = out

        x = self.enc(feats)
        # x = feats

        x_lens = (wavs_lens_rate * x.shape[1]).round().astype(paddle.int64)
        target_lens = target_lens.astype(paddle.int64)
        # target = target.astype(paddle.int32)
        ctc_loss = self.ctc(x, x_lens, target, target_lens)

        return ctc_loss

    @paddle.no_grad()
    def decode(self,
               feats: paddle.Tensor,
               text_feature: Dict[str, int],
               decoding_method: str,
               beam_size: int,
               tokenizer: str=None,
               sb_pipeline=False):
        batch_size = feats.shape[0]

        if decoding_method == 'ctc_prefix_beam_search' and batch_size > 1:
            print(
                f"decoding mode {decoding_method} must be running with batch_size == 1"
            )
            print(f"current batch_size is {batch_size}")

        if decoding_method == 'ctc_greedy_search':
            if tokenizer is None and sb_pipeline is False:
                hyps = self.ctc_greedy_search(feats)
                res = [text_feature.defeaturize(hyp) for hyp in hyps]
                res_tokenids = [hyp for hyp in hyps]
            else:
                if sb_pipeline is True:
                    hyps = self.ctc_greedy_search(feats.unsqueeze(-1))
                else:
                    hyps = self.ctc_greedy_search(feats)
                res = []
                res_tokenids = []
                for sequence in hyps:
                    # Decode token terms to words 
                    predicted_tokens = text_feature.convert_ids_to_tokens(
                        sequence)
                tmp_res = []
                tmp_res_tokenids = []
                for c in predicted_tokens:
                    if c == "[CLS]":
                        continue
                    elif c == "[SEP]" or c == "[PAD]":
                        break
                    else:
                        tmp_res.append(c)
                        tmp_res_tokenids.append(text_feature.vocab[c])
                res.append(''.join(tmp_res))
                res_tokenids.append(tmp_res_tokenids)

        # ctc_prefix_beam_search and attention_rescoring only return one
        # result in List[int], change it to List[List[int]] for compatible
        # with other batch decoding mode
        elif decoding_method == 'ctc_prefix_beam_search':
            assert feats.shape[0] == 1
            if tokenizer is None and sb_pipeline is False:
                hyp = self.ctc_prefix_beam_search(feats, beam_size)
                res = [text_feature.defeaturize(hyp)]
                res_tokenids = [hyp]
            else:
                if sb_pipeline is True:
                    hyp = self.ctc_prefix_beam_search(
                        feats.unsqueeze(-1), beam_size)
                else:
                    hyp = self.ctc_prefix_beam_search(feats, beam_size)
                res = []
                res_tokenids = []
                predicted_tokens = text_feature.convert_ids_to_tokens(hyp)
                tmp_res = []
                tmp_res_tokenids = []
                for c in predicted_tokens:
                    if c == "[CLS]":
                        continue
                    elif c == "[SEP]" or c == "[PAD]":
                        break
                    else:
                        tmp_res.append(c)
                        tmp_res_tokenids.append(text_feature.vocab[c])
                res.append(''.join(tmp_res))
                res_tokenids.append(tmp_res_tokenids)
        else:
            raise ValueError(
                f"WavLM not support decoding method: {decoding_method}")

        return res, res_tokenids

    @classmethod
    def from_config(cls, config):
        model = cls(config)
        return model

    def ctc_greedy_search(self, wav) -> List[List[int]]:
        """ Apply CTC greedy search
        Args:
            speech (paddle.Tensor): (batch, max_len)
            speech_length (paddle.Tensor): (batch, )
        Returns:
            List[List[int]]: best path result
        """
        batch_size = wav.shape[0]
        wav = wav[:, :, 0]
        if self.normalize_wav:
            wav = F.layer_norm(wav, wav.shape[1:])
        # Extract wavlm output
        out = self.wavlm(wav)
        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out, out.shape[1:])
        feats = out
        x = self.enc(feats)
        x_lens = x.shape[1]
        ctc_probs = self.ctc.log_softmax(x)  # (B, maxlen, vocab_size)
        topk_prob, topk_index = ctc_probs.topk(1, axis=2)  # (B, maxlen, 1)
        topk_index = topk_index.reshape([batch_size, x_lens])  # (B, maxlen)

        hyps = [hyp.tolist() for hyp in topk_index]
        hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]
        return hyps

    def _ctc_prefix_beam_search(
            self,
            wav,
            beam_size,
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
        wav = wav[:, :, 0]

        if self.normalize_wav:
            wav = F.layer_norm(wav, wav.shape[1:])
        # Extract wavlm output
        out = self.wavlm(wav)
        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out, out.shape[1:])
        feats = out

        x = self.enc(feats)
        maxlen = x.shape[1]
        ctc_probs = self.ctc.log_softmax(x)  # (1, maxlen, vocab_size)
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
        return hyps

    def ctc_prefix_beam_search(self, wav, beam_size) -> List[int]:
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
        hyps = self._ctc_prefix_beam_search(wav, beam_size)
        return hyps[0][0]


class WavLMBase(nn.Layer):
    """WavLM model"""

    def __init__(self, config: dict):
        super().__init__()
        wavlm_config = WavLMConfig(config)
        wavlm = WavLM(wavlm_config)
        self.wavlm = wavlm

    @classmethod
    def from_config(cls, configs: dict):
        """init model.
        Args:
            configs (dict): config dict.
        Raises:
            ValueError: raise when using not support encoder type.
        Returns:
            nn.Layer: WavLMBase
        """
        model = cls(configs)
        return model

    def forward(self, wav):
        out = self.wavlm(wav)
        return out
