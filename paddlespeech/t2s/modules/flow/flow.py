import logging
import random
from typing import Dict, Optional

import paddle
from omegaconf import DictConfig

from paddlespeech.t2s.models.CosyVoice.mask import make_pad_mask


class MaskedDiffWithXvec(paddle.nn.Layer):
    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 4096,
        input_frame_rate: int = 50,
        only_mask_loss: bool = True,
        encoder: paddle.nn.Layer = None,
        length_regulator: paddle.nn.Layer = None,
        decoder: paddle.nn.Layer = None,
        decoder_conf: Dict = {
            "in_channels": 240,
            "out_channel": 80,
            "spk_emb_dim": 80,
            "n_spks": 1,
            "cfm_params": DictConfig(
                {
                    "sigma_min": 1e-06,
                    "solver": "euler",
                    "t_scheduler": "cosine",
                    "training_cfg_rate": 0.2,
                    "inference_cfg_rate": 0.7,
                    "reg_loss_type": "l1",
                }
            ),
            "decoder_params": {
                "channels": [256, 256],
                "dropout": 0.0,
                "attention_head_dim": 64,
                "n_blocks": 4,
                "num_mid_blocks": 12,
                "num_heads": 8,
                "act_fn": "gelu",
            },
        },
        mel_feat_conf: Dict = {
            "n_fft": 1024,
            "num_mels": 80,
            "sampling_rate": 22050,
            "hop_size": 256,
            "win_size": 1024,
            "fmin": 0,
            "fmax": 8000,
        },
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = paddle.nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = paddle.nn.Linear(
            in_features=spk_embed_dim, out_features=output_size
        )
        self.encoder = encoder
        self.encoder_proj = paddle.nn.Linear(
            in_features=self.encoder.output_size(), out_features=output_size
        )
        self.decoder = decoder
        self.length_regulator = length_regulator
        self.only_mask_loss = only_mask_loss

    def forward(
        self, batch: dict, device: paddle.device
    ) -> Dict[str, Optional[paddle.Tensor]]:
        token = batch["speech_token"].to(device)
        token_len = batch["speech_token_len"].to(device)
        feat = batch["speech_feat"].to(device)
        feat_len = batch["speech_feat_len"].to(device)
        embedding = batch["embedding"].to(device)
        embedding = paddle.nn.functional.normalize(x=embedding, axis=1)
        embedding = self.spk_embed_affine_layer(embedding)
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(paddle.clip(token, min=0)) * mask
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        h, h_lengths = self.length_regulator(h, feat_len)
        conds = paddle.zeros(feat.shape, device=token.place)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = paddle.transpose(conds, perm=[0, 2, 1])
        mask = (~make_pad_mask(feat_len)).to(h)
        loss, _ = self.decoder.compute_loss(
            paddle.transpose(feat, perm=[0, 2, 1]),
            mask.unsqueeze(1),
            paddle.transpose(h, perm=[0, 2, 1]),
            embedding,
            cond=conds,
        )
        return {"loss": loss}

    @paddle.no_grad()
    def inference(
        self,
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        prompt_feat_len,
        embedding,
        flow_cache,
    ):
        assert token.shape[0] == 1
        embedding = paddle.nn.functional.normalize(x=embedding, axis=1)
        embedding = self.spk_embed_affine_layer(embedding)
        token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
        token, token_len = (
            paddle.cat([prompt_token, token], dim=1),
            prompt_token_len + token_len,
        )
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(paddle.clip(token, min=0)) * mask
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        mel_len1, mel_len2 = prompt_feat.shape[1], int(
            token_len2 / self.input_frame_rate * 22050 / 256
        )
        h, h_lengths = self.length_regulator.inference(
            h[:, :token_len1],
            h[:, token_len1:],
            mel_len1,
            mel_len2,
            self.input_frame_rate,
        )
        conds = paddle.zeros(
            [1, mel_len1 + mel_len2, self.output_size], device=token.place
        ).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = paddle.transpose(conds, perm=[0, 2, 1])
        
        mask = (~make_pad_mask(paddle.tensor([mel_len1 + mel_len2]))).to(h)
        feat, flow_cache = self.decoder(
            mu=paddle.transpose(h, perm=[0, 2, 1]),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            prompt_len=mel_len1,
            cache=flow_cache,
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), flow_cache


class CausalMaskedDiffWithXvec(paddle.nn.Layer):
    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 6561,
        input_frame_rate: int = 25,
        only_mask_loss: bool = True,
        token_mel_ratio: int = 2,
        pre_lookahead_len: int = 3,
        encoder: paddle.nn.Layer = None,
        decoder: paddle.nn.Layer = None,
        decoder_conf: Dict = {
            "in_channels": 240,
            "out_channel": 80,
            "spk_emb_dim": 80,
            "n_spks": 1,
            "cfm_params": DictConfig(
                {
                    "sigma_min": 1e-06,
                    "solver": "euler",
                    "t_scheduler": "cosine",
                    "training_cfg_rate": 0.2,
                    "inference_cfg_rate": 0.7,
                    "reg_loss_type": "l1",
                }
            ),
            "decoder_params": {
                "channels": [256, 256],
                "dropout": 0.0,
                "attention_head_dim": 64,
                "n_blocks": 4,
                "num_mid_blocks": 12,
                "num_heads": 8,
                "act_fn": "gelu",
            },
        },
        mel_feat_conf: Dict = {
            "n_fft": 1024,
            "num_mels": 80,
            "sampling_rate": 22050,
            "hop_size": 256,
            "win_size": 1024,
            "fmin": 0,
            "fmax": 8000,
        },
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = paddle.nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = paddle.nn.Linear(
            in_features=spk_embed_dim, out_features=output_size
        )
        self.encoder = encoder
        self.encoder_proj = paddle.nn.Linear(
            in_features=self.encoder.output_size(), out_features=output_size
        )
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len

    def forward(
        self, batch: dict, device: paddle.device
    ) -> Dict[str, Optional[paddle.Tensor]]:
        token = batch["speech_token"].to(device)
        token_len = batch["speech_token_len"].to(device)
        feat = batch["speech_feat"].to(device)
        feat_len = batch["speech_feat_len"].to(device)
        embedding = batch["embedding"].to(device)
        streaming = True if random.random() < 0.5 else False
        embedding = paddle.nn.functional.normalize(x=embedding, axis=1)
        embedding = self.spk_embed_affine_layer(embedding)
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(paddle.clip(token, min=0)) * mask
        h, h_lengths = self.encoder(token, token_len, streaming=streaming)
        h = self.encoder_proj(h)
        conds = paddle.zeros(feat.shape, device=token.place)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = paddle.transpose(conds, perm=[0, 2, 1])
        
        mask = (~make_pad_mask(h_lengths.sum(dim=-1).squeeze(dim=1))).to(h)
        loss, _ = self.decoder.compute_loss(
            paddle.transpose(feat, perm=[0, 2, 1]).contiguous(),
            mask.unsqueeze(1),
            paddle.transpose(h, perm=[0, 2, 1]).contiguous(),
            embedding,
            cond=conds,
            streaming=streaming,
        )
        return {"loss": loss}

    @paddle.no_grad()
    def inference(
        self,
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        prompt_feat_len,
        embedding,
        streaming,
        finalize,
    ):
        assert token.shape[0] == 1
        embedding = paddle.nn.functional.normalize(x=embedding, axis=1)
        embedding = self.spk_embed_affine_layer(embedding)

        token, token_len = (
            paddle.cat([prompt_token, token], dim=1),
            prompt_token_len + token_len,
        )
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(paddle.clip(token, min=0)) * mask
        if finalize is True:
            h, h_lengths = self.encoder(token, token_len, streaming=streaming)
        else:
            token, context = (
                token[:, : -self.pre_lookahead_len],
                token[:, -self.pre_lookahead_len :],
            )
            h, h_lengths = self.encoder(
                token, token_len, context=context, streaming=streaming
            )
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
        h = self.encoder_proj(h)
        conds = paddle.zeros(
            [1, mel_len1 + mel_len2, self.output_size]
        ).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = paddle.transpose(conds, perm=[0, 2, 1])
        mask = (~make_pad_mask(paddle.to_tensor([mel_len1 + mel_len2],dtype='int32'))).to(h)
        feat, _ = self.decoder(
            mu=paddle.transpose(h, perm=[0, 2, 1]).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            streaming=streaming,
        )
        paddle.save(feat,'/root/paddlejob/workspace/zhangjinghong/CosyVoice/feat.pdparams')
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), None
