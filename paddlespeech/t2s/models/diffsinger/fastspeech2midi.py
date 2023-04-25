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
# Modified from espnet(https://github.com/espnet/espnet)
from typing import Any
from typing import Dict
from typing import Sequence
from typing import Tuple

import paddle
from paddle import nn
from typeguard import check_argument_types

from paddlespeech.t2s.models.fastspeech2 import FastSpeech2
from paddlespeech.t2s.models.fastspeech2 import FastSpeech2Loss
from paddlespeech.t2s.modules.losses import ssim
from paddlespeech.t2s.modules.masked_fill import masked_fill
from paddlespeech.t2s.modules.nets_utils import make_non_pad_mask
from paddlespeech.t2s.modules.nets_utils import make_pad_mask


class FastSpeech2MIDI(FastSpeech2):
    """The Fastspeech2 module of DiffSinger.
    """

    def __init__(
            self,
            # fastspeech2 network structure related
            idim: int,
            odim: int,
            fastspeech2_params: Dict[str, Any],
            # note emb
            note_num: int=300,
            # is_slur emb
            is_slur_num: int=2,
            use_energy_pred: bool=False,
            use_postnet: bool=False, ):
        """Initialize FastSpeech2 module for svs.
        Args:
            fastspeech2_params (Dict):
                The config of FastSpeech2 module on DiffSinger model
            note_num (Optional[int]): 
                Number of note. If not None, assume that the
                note_ids will be provided as the input and use note_embedding_table.
            is_slur_num (Optional[int]): 
                Number of note. If not None, assume that the
                is_slur_ids will be provided as the input
    
        """
        assert check_argument_types()
        super().__init__(idim=idim, odim=odim, **fastspeech2_params)
        self.use_energy_pred = use_energy_pred
        self.use_postnet = use_postnet
        if not self.use_postnet:
            self.postnet = None

        self.note_embed_dim = self.is_slur_embed_dim = fastspeech2_params[
            "adim"]

        # note_ embed
        self.note_embedding_table = nn.Embedding(
            num_embeddings=note_num,
            embedding_dim=self.note_embed_dim,
            padding_idx=self.padding_idx)
        self.note_dur_layer = nn.Linear(1, self.note_embed_dim)

        # slur embed
        self.is_slur_embedding_table = nn.Embedding(
            num_embeddings=is_slur_num,
            embedding_dim=self.is_slur_embed_dim,
            padding_idx=self.padding_idx)

    def forward(
            self,
            text: paddle.Tensor,
            note: paddle.Tensor,
            note_dur: paddle.Tensor,
            is_slur: paddle.Tensor,
            text_lengths: paddle.Tensor,
            speech: paddle.Tensor,
            speech_lengths: paddle.Tensor,
            durations: paddle.Tensor,
            pitch: paddle.Tensor,
            energy: paddle.Tensor,
            spk_emb: paddle.Tensor=None,
            spk_id: paddle.Tensor=None,
    ) -> Tuple[paddle.Tensor, Dict[str, paddle.Tensor], paddle.Tensor]:
        """Calculate forward propagation.

        Args:
            text(Tensor(int64)): 
                Batch of padded token (phone) ids (B, Tmax).
            note(Tensor(int64)): 
                Batch of padded note (element in music score) ids (B, Tmax).
            note_dur(Tensor(float32)): 
                Batch of padded note durations in seconds (element in music score) (B, Tmax).
            is_slur(Tensor(int64)): 
                Batch of padded slur (element in music score) ids (B, Tmax).
            text_lengths(Tensor(int64)): 
                Batch of phone lengths of each input (B,).
            speech(Tensor[float32]): 
                Batch of padded target features (e.g. mel) (B, Lmax, odim).
            speech_lengths(Tensor(int64)): 
                Batch of the lengths of each target features (B,).
            durations(Tensor(int64)): 
                Batch of padded token durations in frame (B, Tmax).
            pitch(Tensor[float32]): 
                Batch of padded frame-averaged pitch (B, Lmax, 1).
            energy(Tensor[float32]): 
                Batch of padded frame-averaged energy (B, Lmax, 1).
            spk_emb(Tensor[float32], optional): 
                Batch of speaker embeddings (B, spk_embed_dim).
            spk_id(Tnesor[int64], optional(int64)): 
                Batch of speaker ids (B,)

        Returns:

        """
        xs = paddle.cast(text, 'int64')
        note = paddle.cast(note, 'int64')
        note_dur = paddle.cast(note_dur, 'float32')
        is_slur = paddle.cast(is_slur, 'int64')
        ilens = paddle.cast(text_lengths, 'int64')
        olens = paddle.cast(speech_lengths, 'int64')
        ds = paddle.cast(durations, 'int64')
        ps = pitch
        es = energy
        ys = speech
        olens = speech_lengths
        if spk_id is not None:
            spk_id = paddle.cast(spk_id, 'int64')
        # forward propagation
        before_outs, after_outs, d_outs, p_outs, e_outs, spk_logits = self._forward(
            xs=xs,
            note=note,
            note_dur=note_dur,
            is_slur=is_slur,
            ilens=ilens,
            olens=olens,
            ds=ds,
            ps=ps,
            es=es,
            is_inference=False,
            spk_emb=spk_emb,
            spk_id=spk_id, )
        # modify mod part of groundtruth
        if self.reduction_factor > 1:
            olens = olens - olens % self.reduction_factor
            max_olen = max(olens)
            ys = ys[:, :max_olen]

        return before_outs, after_outs, d_outs, p_outs, e_outs, ys, olens, spk_logits

    def _forward(
            self,
            xs: paddle.Tensor,
            note: paddle.Tensor,
            note_dur: paddle.Tensor,
            is_slur: paddle.Tensor,
            ilens: paddle.Tensor,
            olens: paddle.Tensor=None,
            ds: paddle.Tensor=None,
            ps: paddle.Tensor=None,
            es: paddle.Tensor=None,
            is_inference: bool=False,
            is_train_diffusion: bool=False,
            return_after_enc=False,
            alpha: float=1.0,
            spk_emb=None,
            spk_id=None, ) -> Sequence[paddle.Tensor]:

        before_outs = after_outs = d_outs = p_outs = e_outs = spk_logits = None
        # forward encoder
        masks = self._source_mask(ilens)
        note_emb = self.note_embedding_table(note)
        note_dur_emb = self.note_dur_layer(paddle.unsqueeze(note_dur, axis=-1))
        is_slur_emb = self.is_slur_embedding_table(is_slur)

        # (B, Tmax, adim)
        hs, _ = self.encoder(
            xs=xs,
            masks=masks,
            note_emb=note_emb,
            note_dur_emb=note_dur_emb,
            is_slur_emb=is_slur_emb, )

        if self.spk_num and self.enable_speaker_classifier and not is_inference:
            hs_for_spk_cls = self.grad_reverse(hs)
            spk_logits = self.speaker_classifier(hs_for_spk_cls, ilens)
        else:
            spk_logits = None

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            # spk_emb has a higher priority than spk_id
            if spk_emb is not None:
                hs = self._integrate_with_spk_embed(hs, spk_emb)
            elif spk_id is not None:
                spk_emb = self.spk_embedding_table(spk_id)
                hs = self._integrate_with_spk_embed(hs, spk_emb)

        # forward duration predictor (phone-level) and variance predictors (frame-level)
        d_masks = make_pad_mask(ilens)
        if olens is not None:
            pitch_masks = make_pad_mask(olens).unsqueeze(-1)
        else:
            pitch_masks = None

        # inference for decoder input for diffusion
        if is_train_diffusion:
            hs = self.length_regulator(hs, ds, is_inference=False)
            p_outs = self.pitch_predictor(hs.detach(), pitch_masks)
            p_embs = self.pitch_embed(p_outs.transpose((0, 2, 1))).transpose(
                (0, 2, 1))
            hs += p_embs
            if self.use_energy_pred:
                e_outs = self.energy_predictor(hs.detach(), pitch_masks)
                e_embs = self.energy_embed(
                    e_outs.transpose((0, 2, 1))).transpose((0, 2, 1))
                hs += e_embs

        elif is_inference:
            # (B, Tmax)
            if ds is not None:
                d_outs = ds
            else:
                d_outs = self.duration_predictor.inference(hs, d_masks)

            # (B, Lmax, adim)
            hs = self.length_regulator(hs, d_outs, alpha, is_inference=True)

            if ps is not None:
                p_outs = ps
            else:
                if self.stop_gradient_from_pitch_predictor:
                    p_outs = self.pitch_predictor(hs.detach(), pitch_masks)
                else:
                    p_outs = self.pitch_predictor(hs, pitch_masks)
            p_embs = self.pitch_embed(p_outs.transpose((0, 2, 1))).transpose(
                (0, 2, 1))
            hs += p_embs

            if self.use_energy_pred:
                if es is not None:
                    e_outs = es
                else:
                    if self.stop_gradient_from_energy_predictor:
                        e_outs = self.energy_predictor(hs.detach(), pitch_masks)
                    else:
                        e_outs = self.energy_predictor(hs, pitch_masks)
                e_embs = self.energy_embed(
                    e_outs.transpose((0, 2, 1))).transpose((0, 2, 1))
                hs += e_embs

        # training
        else:
            d_outs = self.duration_predictor(hs, d_masks)
            # (B, Lmax, adim)
            hs = self.length_regulator(hs, ds, is_inference=False)
            if self.stop_gradient_from_pitch_predictor:
                p_outs = self.pitch_predictor(hs.detach(), pitch_masks)
            else:
                p_outs = self.pitch_predictor(hs, pitch_masks)
            p_embs = self.pitch_embed(ps.transpose((0, 2, 1))).transpose(
                (0, 2, 1))
            hs += p_embs

            if self.use_energy_pred:
                if self.stop_gradient_from_energy_predictor:
                    e_outs = self.energy_predictor(hs.detach(), pitch_masks)
                else:
                    e_outs = self.energy_predictor(hs, pitch_masks)
                e_embs = self.energy_embed(es.transpose((0, 2, 1))).transpose(
                    (0, 2, 1))
                hs += e_embs

        # forward decoder
        if olens is not None and not is_inference:
            if self.reduction_factor > 1:
                olens_in = paddle.to_tensor(
                    [olen // self.reduction_factor for olen in olens.numpy()])
            else:
                olens_in = olens
            # (B, 1, T)
            h_masks = self._source_mask(olens_in)
        else:
            h_masks = None

        if return_after_enc:
            return hs, h_masks

        if self.decoder_type == 'cnndecoder':
            # remove output masks for dygraph to static graph
            zs = self.decoder(hs, h_masks)
            before_outs = zs
        else:
            # (B, Lmax, adim)
            zs, _ = self.decoder(hs, h_masks)
            # (B, Lmax, odim)
            before_outs = self.feat_out(zs).reshape(
                (paddle.shape(zs)[0:1], -1, self.odim))

        # postnet -> (B, Lmax//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose((0, 2, 1))).transpose((0, 2, 1))

        return before_outs, after_outs, d_outs, p_outs, e_outs, spk_logits

    def encoder_infer(
            self,
            text: paddle.Tensor,
            note: paddle.Tensor,
            note_dur: paddle.Tensor,
            is_slur: paddle.Tensor,
            alpha: float=1.0,
            spk_emb=None,
            spk_id=None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        xs = paddle.cast(text, 'int64').unsqueeze(0)
        note = paddle.cast(note, 'int64').unsqueeze(0)
        note_dur = paddle.cast(note_dur, 'float32').unsqueeze(0)
        is_slur = paddle.cast(is_slur, 'int64').unsqueeze(0)
        # setup batch axis
        ilens = paddle.shape(xs)[1:2]

        if spk_emb is not None:
            spk_emb = spk_emb.unsqueeze(0)

        # (1, L, odim)
        # use *_ to avoid bug in dygraph to static graph    
        hs, _ = self._forward(
            xs=xs,
            note=note,
            note_dur=note_dur,
            is_slur=is_slur,
            ilens=ilens,
            is_inference=True,
            return_after_enc=True,
            alpha=alpha,
            spk_emb=spk_emb,
            spk_id=spk_id, )
        return hs

    # get encoder output for diffusion training
    def encoder_infer_batch(
            self,
            text: paddle.Tensor,
            note: paddle.Tensor,
            note_dur: paddle.Tensor,
            is_slur: paddle.Tensor,
            text_lengths: paddle.Tensor,
            speech_lengths: paddle.Tensor,
            ds: paddle.Tensor=None,
            ps: paddle.Tensor=None,
            es: paddle.Tensor=None,
            alpha: float=1.0,
            spk_emb=None,
            spk_id=None, ) -> Tuple[paddle.Tensor, paddle.Tensor]:

        xs = paddle.cast(text, 'int64')
        note = paddle.cast(note, 'int64')
        note_dur = paddle.cast(note_dur, 'float32')
        is_slur = paddle.cast(is_slur, 'int64')
        ilens = paddle.cast(text_lengths, 'int64')
        olens = paddle.cast(speech_lengths, 'int64')

        if spk_emb is not None:
            spk_emb = spk_emb.unsqueeze(0)

        # (1, L, odim)
        # use *_ to avoid bug in dygraph to static graph    
        hs, h_masks = self._forward(
            xs=xs,
            note=note,
            note_dur=note_dur,
            is_slur=is_slur,
            ilens=ilens,
            olens=olens,
            ds=ds,
            ps=ps,
            es=es,
            return_after_enc=True,
            is_train_diffusion=True,
            alpha=alpha,
            spk_emb=spk_emb,
            spk_id=spk_id, )
        return hs, h_masks

    def inference(
            self,
            text: paddle.Tensor,
            note: paddle.Tensor,
            note_dur: paddle.Tensor,
            is_slur: paddle.Tensor,
            durations: paddle.Tensor=None,
            pitch: paddle.Tensor=None,
            energy: paddle.Tensor=None,
            alpha: float=1.0,
            use_teacher_forcing: bool=False,
            spk_emb=None,
            spk_id=None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Generate the sequence of features given the sequences of characters.

        Args:
            text(Tensor(int64)): 
                Input sequence of characters (T,).
            note(Tensor(int64)): 
                Input note (element in music score) ids (T,).
            note_dur(Tensor(float32)): 
               Input note durations in seconds (element in music score) (T,).
            is_slur(Tensor(int64)): 
                Input slur (element in music score) ids (T,).
            durations(Tensor, optional (int64)): 
                Groundtruth of duration (T,).
            pitch(Tensor, optional): 
                Groundtruth of token-averaged pitch (T, 1).
            energy(Tensor, optional): 
                Groundtruth of token-averaged energy (T, 1).
            alpha(float, optional): 
                Alpha to control the speed.
            use_teacher_forcing(bool, optional): 
                Whether to use teacher forcing.
                If true, groundtruth of duration, pitch and energy will be used.
            spk_emb(Tensor, optional, optional): 
                peaker embedding vector (spk_embed_dim,). (Default value = None)
            spk_id(Tensor, optional(int64), optional): 
                spk ids (1,). (Default value = None)

        Returns:

        """
        xs = paddle.cast(text, 'int64').unsqueeze(0)
        note = paddle.cast(note, 'int64').unsqueeze(0)
        note_dur = paddle.cast(note_dur, 'float32').unsqueeze(0)
        is_slur = paddle.cast(is_slur, 'int64').unsqueeze(0)
        d, p, e = durations, pitch, energy
        # setup batch axis
        ilens = paddle.shape(xs)[1:2]

        if spk_emb is not None:
            spk_emb = spk_emb.unsqueeze(0)

        if use_teacher_forcing:
            # use groundtruth of duration, pitch, and energy
            ds = d.unsqueeze(0) if d is not None else None
            ps = p.unsqueeze(0) if p is not None else None
            es = e.unsqueeze(0) if e is not None else None

            # (1, L, odim)
            _, outs, d_outs, p_outs, e_outs, _ = self._forward(
                xs=xs,
                note=note,
                note_dur=note_dur,
                is_slur=is_slur,
                ilens=ilens,
                ds=ds,
                ps=ps,
                es=es,
                spk_emb=spk_emb,
                spk_id=spk_id,
                is_inference=True)
        else:
            # (1, L, odim)
            _, outs, d_outs, p_outs, e_outs, _ = self._forward(
                xs=xs,
                note=note,
                note_dur=note_dur,
                is_slur=is_slur,
                ilens=ilens,
                is_inference=True,
                alpha=alpha,
                spk_emb=spk_emb,
                spk_id=spk_id, )

        if e_outs is None:
            e_outs = [None]

        return outs[0], d_outs[0], p_outs[0], e_outs[0]


class FastSpeech2MIDILoss(FastSpeech2Loss):
    """Loss function module for DiffSinger."""

    def __init__(self, use_masking: bool=True,
                 use_weighted_masking: bool=False):
        """Initialize feed-forward Transformer loss module.
        Args:
            use_masking (bool): 
                Whether to apply masking for padded part in loss calculation.
            use_weighted_masking (bool): 
                Whether to weighted masking in loss calculation.
        """
        assert check_argument_types()
        super().__init__(use_masking, use_weighted_masking)

    def forward(
            self,
            after_outs: paddle.Tensor,
            before_outs: paddle.Tensor,
            d_outs: paddle.Tensor,
            p_outs: paddle.Tensor,
            e_outs: paddle.Tensor,
            ys: paddle.Tensor,
            ds: paddle.Tensor,
            ps: paddle.Tensor,
            es: paddle.Tensor,
            ilens: paddle.Tensor,
            olens: paddle.Tensor,
            spk_logits: paddle.Tensor=None,
            spk_ids: paddle.Tensor=None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor,
               paddle.Tensor, ]:
        """Calculate forward propagation.

        Args:
            after_outs(Tensor):  
                Batch of outputs after postnets (B, Lmax, odim).
            before_outs(Tensor): 
                Batch of outputs before postnets (B, Lmax, odim).
            d_outs(Tensor): 
                Batch of outputs of duration predictor (B, Tmax).
            p_outs(Tensor): 
                Batch of outputs of pitch predictor (B, Lmax, 1).
            e_outs(Tensor): 
                Batch of outputs of energy predictor (B, Lmax, 1).
            ys(Tensor): 
                Batch of target features (B, Lmax, odim).
            ds(Tensor): 
                Batch of durations (B, Tmax).
            ps(Tensor): 
                Batch of target frame-averaged pitch (B, Lmax, 1).
            es(Tensor): 
                Batch of target frame-averaged energy (B, Lmax, 1).
            ilens(Tensor): 
                Batch of the lengths of each input (B,).
            olens(Tensor): 
                Batch of the lengths of each target (B,).
            spk_logits(Option[Tensor]):
                Batch of outputs after speaker classifier (B, Lmax, num_spk)
            spk_ids(Option[Tensor]):
                Batch of target spk_id (B,)
        

        Returns:

        
        """
        l1_loss = duration_loss = pitch_loss = energy_loss = speaker_loss = ssim_loss = 0.0

        # apply mask to remove padded part
        if self.use_masking:
            # make feature for ssim loss
            out_pad_masks = make_pad_mask(olens).unsqueeze(-1)
            before_outs_ssim = masked_fill(before_outs, out_pad_masks, 0.0)
            if not paddle.equal_all(after_outs, before_outs):
                after_outs_ssim = masked_fill(after_outs, out_pad_masks, 0.0)
            ys_ssim = masked_fill(ys, out_pad_masks, 0.0)

            out_masks = make_non_pad_mask(olens).unsqueeze(-1)
            before_outs = before_outs.masked_select(
                out_masks.broadcast_to(before_outs.shape))
            if not paddle.equal_all(after_outs, before_outs):
                after_outs = after_outs.masked_select(
                    out_masks.broadcast_to(after_outs.shape))
            ys = ys.masked_select(out_masks.broadcast_to(ys.shape))
            duration_masks = make_non_pad_mask(ilens)
            d_outs = d_outs.masked_select(
                duration_masks.broadcast_to(d_outs.shape))
            ds = ds.masked_select(duration_masks.broadcast_to(ds.shape))
            pitch_masks = out_masks
            p_outs = p_outs.masked_select(
                pitch_masks.broadcast_to(p_outs.shape))
            ps = ps.masked_select(pitch_masks.broadcast_to(ps.shape))
            if e_outs is not None:
                e_outs = e_outs.masked_select(
                    pitch_masks.broadcast_to(e_outs.shape))
                es = es.masked_select(pitch_masks.broadcast_to(es.shape))

            if spk_logits is not None and spk_ids is not None:
                batch_size = spk_ids.shape[0]
                spk_ids = paddle.repeat_interleave(spk_ids, spk_logits.shape[1],
                                                   None)
                spk_logits = paddle.reshape(spk_logits,
                                            [-1, spk_logits.shape[-1]])
                mask_index = spk_logits.abs().sum(axis=1) != 0
                spk_ids = spk_ids[mask_index]
                spk_logits = spk_logits[mask_index]

        # calculate loss
        l1_loss = self.l1_criterion(before_outs, ys)
        ssim_loss = 1.0 - ssim(
            before_outs_ssim.unsqueeze(1), ys_ssim.unsqueeze(1))
        if not paddle.equal_all(after_outs, before_outs):
            l1_loss += self.l1_criterion(after_outs, ys)
            ssim_loss += (
                1.0 - ssim(after_outs_ssim.unsqueeze(1), ys_ssim.unsqueeze(1)))
        l1_loss = l1_loss * 0.5
        ssim_loss = ssim_loss * 0.5

        duration_loss = self.duration_criterion(d_outs, ds)
        pitch_loss = self.l1_criterion(p_outs, ps)
        if e_outs is not None:
            energy_loss = self.l1_criterion(e_outs, es)

        if spk_logits is not None and spk_ids is not None:
            speaker_loss = self.ce_criterion(spk_logits, spk_ids) / batch_size

        # make weighted mask and apply it
        if self.use_weighted_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1)
            out_weights = out_masks.cast(dtype=paddle.float32) / out_masks.cast(
                dtype=paddle.float32).sum(
                    axis=1, keepdim=True)
            out_weights /= ys.shape[0] * ys.shape[2]
            duration_masks = make_non_pad_mask(ilens)
            duration_weights = (duration_masks.cast(dtype=paddle.float32) /
                                duration_masks.cast(dtype=paddle.float32).sum(
                                    axis=1, keepdim=True))
            duration_weights /= ds.shape[0]

            # apply weight
            l1_loss = l1_loss.multiply(out_weights)
            l1_loss = l1_loss.masked_select(
                out_masks.broadcast_to(l1_loss.shape)).sum()
            ssim_loss = ssim_loss.multiply(out_weights)
            ssim_loss = ssim_loss.masked_select(
                out_masks.broadcast_to(ssim_loss.shape)).sum()
            duration_loss = (duration_loss.multiply(duration_weights)
                             .masked_select(duration_masks).sum())
            pitch_masks = out_masks
            pitch_weights = out_weights
            pitch_loss = pitch_loss.multiply(pitch_weights)
            pitch_loss = pitch_loss.masked_select(
                pitch_masks.broadcast_to(pitch_loss.shape)).sum()
            if e_outs is not None:
                energy_loss = energy_loss.multiply(pitch_weights)
                energy_loss = energy_loss.masked_select(
                    pitch_masks.broadcast_to(energy_loss.shape)).sum()

        return l1_loss, ssim_loss, duration_loss, pitch_loss, energy_loss, speaker_loss
