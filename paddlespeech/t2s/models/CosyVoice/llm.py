import queue
import random
import threading
import time
from typing import Callable, Dict, Generator, List, Optional
import logging
import paddle.nn.functional as F
import paddle
IGNORE_ID = -1
# from cosyvoice.transformer.label_smoothing_loss import LabelSmoothingLoss
# from cosyvoice.utils.common import IGNORE_ID, th_accuracy
# from cosyvoice.utils.file_utils import logging
# from cosyvoice.utils.mask import make_pad_mask
import torch
LabelSmoothingLoss = None
def ras_sampling(weighted_scores, decoded_tokens, sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1):
    top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
    recent_tokens = paddle.to_tensor(decoded_tokens[-win_size:], dtype='int64')
    rep_num = paddle.sum(recent_tokens.cpu() == top_ids.cpu()).cpu().item()
    if rep_num >= win_size * tau_r:
        top_ids = random_sampling(weighted_scores, decoded_tokens, sampling)
    return top_ids


def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25):
    softmax_scores = paddle.nn.functional.softmax(weighted_scores, axis=0)
    sorted_indices = paddle.argsort(softmax_scores, axis=0, descending=True)
    sorted_probs = paddle.gather(softmax_scores, sorted_indices, axis=0)
    
    prob_list = []
    indices_list = []
    cum_prob = 0.0
    
    for i in range(len(sorted_indices)):
        if cum_prob < top_p and len(prob_list) < top_k:
            cum_prob += sorted_probs[i].item()
            prob_list.append(sorted_probs[i])
            indices_list.append(sorted_indices[i])
        else:
            break
    
    prob_tensor = paddle.to_tensor(prob_list, dtype=weighted_scores.dtype)
    indices_tensor = paddle.to_tensor(indices_list, dtype='int64')
    top_ids = indices_tensor[paddle.multinomial(prob_tensor, num_samples=1, replacement=True)]
    
    return top_ids


def random_sampling(weighted_scores, decoded_tokens, sampling):
    probs = paddle.nn.functional.softmax(weighted_scores, axis=0)
    top_ids = paddle.multinomial(probs, num_samples=1, replacement=True)
    return top_ids
def make_pad_mask(lengths: paddle.Tensor, max_len: int = 0) -> paddle.Tensor:
    batch_size = lengths.shape[0]
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = paddle.arange(0, max_len, dtype='int64') 
    seq_range_expand = seq_range.unsqueeze(0).expand([batch_size, max_len])
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask

def th_accuracy(pad_outputs: paddle.Tensor, pad_targets: paddle.Tensor,
                ignore_label: int) -> paddle.Tensor:
    pad_pred = pad_outputs.reshape((pad_targets.shape[0], pad_targets.shape[1], -1)).argmax(axis=2)
    mask = pad_targets != ignore_label
    numerator = paddle.sum((pad_pred[mask] == pad_targets[mask]).astype('float32'))
    denominator = paddle.sum(mask.astype('float32'))
    accuracy = numerator / denominator
    
    return accuracy.detach()
class TransformerLM(paddle.nn.Layer):
    def __init__(
        self,
        text_encoder_input_size: int,
        llm_input_size: int,
        llm_output_size: int,
        text_token_size: int,
        speech_token_size: int,
        text_encoder: paddle.nn.Layer,
        llm: paddle.nn.Layer,
        sampling: Callable,
        length_normalized_loss: bool = True,
        lsm_weight: float = 0.0,
        spk_embed_dim: int = 192,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.speech_token_size = speech_token_size
        self.text_embedding = paddle.nn.Embedding(
            text_token_size, text_encoder_input_size
        )
        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = paddle.nn.Linear(
            in_features=self.text_encoder.output_size(), out_features=llm_input_size
        )
        self.sos_eos = 0
        self.task_id = 1
        self.llm_embedding = paddle.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = paddle.nn.Linear(
            in_features=llm_output_size, out_features=speech_token_size + 1
        )
        
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 1,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        self.speech_embedding = paddle.nn.Embedding(speech_token_size, llm_input_size)
        self.spk_embed_affine_layer = paddle.nn.Linear(
            in_features=spk_embed_dim, out_features=llm_input_size
        )
        self.sampling = sampling

    def encode(self, text: paddle.Tensor, text_lengths: paddle.Tensor):
        encoder_out, encoder_mask = self.text_encoder(
            text, text_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1
        )
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(
        self,
        sos_eos_emb,
        embedding,
        text_token,
        text_token_len,
        task_id_emb,
        speech_token,
        speech_token_len,
    ):

        text_token = paddle.static.nn.sequence_unpad(
            text_token, text_token_len.cpu()
        )
        speech_token = paddle.static.nn.sequence_unpad(
            speech_token, speech_token_len.cpu()
        )
        lm_input = [
            paddle.cat(
                [
                    sos_eos_emb.squeeze(dim=0),
                    embedding[i],
                    text_token[i],
                    task_id_emb.squeeze(dim=0),
                    speech_token[i],
                ],
                dim=0,
            )
            for i in range(len(text_token))
        ]
        lm_input_len = paddle.tensor([i.size(0) for i in lm_input], dtype=paddle.int32)
        lm_input = paddle.static.nn.sequence_unpad(
            lm_input, batch_first=True, padding_value=IGNORE_ID
        )
        return lm_input, lm_input_len

    def forward(
        self, batch: dict, device: torch.device
    ) -> Dict[str, Optional[paddle.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch["text_token"].to(device)
        text_token_len = batch["text_token_len"].to(device)
        speech_token = batch["speech_token"].to(device)
        speech_token_len = batch["speech_token_len"].to(device)
        embedding = batch["embedding"].to(device)
        lm_target = [
            paddle.tensor(
                [IGNORE_ID] * (2 + text_token_len[i])
                + speech_token[i, : speech_token_len[i]].tolist()
                + [self.speech_token_size]
            )
            for i in range(text_token.size(0))
        ]
        lm_target = torch.nn.utils.rnn.pad_sequence(
            lm_target, batch_first=True, padding_value=IGNORE_ID
        ).to(device)
        text_token = self.text_embedding(text_token)
        text_token, text_token_len = self.encode(text_token, text_token_len)
        embedding = paddle.nn.functional.normalize(x=embedding, axis=1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1)
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        speech_token = self.speech_embedding(speech_token)
        lm_input, lm_input_len = self.pad_unpad_sequence(
            sos_eos_emb,
            embedding,
            text_token,
            text_token_len,
            task_id_emb,
            speech_token,
            speech_token_len,
        )
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(
            logits.view(-1, self.speech_token_size + 1),
            lm_target,
            ignore_label=IGNORE_ID,
        )
        return {"loss": loss, "acc": acc}

    def sampling_ids(
        self,
        weighted_scores: paddle.Tensor,
        decoded_tokens: List,
        sampling: int,
        ignore_eos: bool = True,
    ):
        num_trials, max_trials = 0, 100
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if not ignore_eos or self.speech_token_size not in top_ids:
                break
            num_trials += 1
            if num_trials > max_trials:
                raise RuntimeError(
                    "sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!".format(
                        max_trials
                    )
                )
        return top_ids

    @paddle.no_grad()
    def inference(
        self,
        text: paddle.Tensor,
        text_len: paddle.Tensor,
        prompt_text: paddle.Tensor,
        prompt_text_len: paddle.Tensor,
        prompt_speech_token: paddle.Tensor,
        prompt_speech_token_len: paddle.Tensor,
        embedding: paddle.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
        uuid: str = "",
    ) -> Generator[paddle.Tensor, None, None]:
        device = text.place
        text = paddle.cat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.text_embedding(text)
        text, text_len = self.encode(text, text_len)
        if embedding.shape[0] != 0:
            embedding = paddle.nn.functional.normalize(x=embedding, axis=1)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = embedding.unsqueeze(dim=1)
        else:
            embedding = (
                paddle.zeros(1, 0, self.llm_input_size, dtype=text.dtype)
                .to(device)
                .to(text.dtype)
            )
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = paddle.zeros(
                1, 0, self.llm_input_size, dtype=text.dtype
            ).to(device)
        lm_input = paddle.cat(
            [sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1
        )
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)
        out_tokens = []
        offset = 0
        att_cache, cnn_cache = paddle.zeros(
            (0, 0, 0, 0), device=lm_input.place
        ), paddle.zeros((0, 0, 0, 0), device=lm_input.place)
        for i in range(max_len):
            y_pred, att_cache, cnn_cache = self.llm.forward_chunk(
                lm_input,
                offset=offset,
                required_cache_size=-1,
                att_cache=att_cache,
                cnn_cache=cnn_cache,
                att_mask=paddle.tril(
                    paddle.ones(
                        (1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.place
                    )
                ).to(paddle.bool),
            )
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            if i == 0:
                logp[:, self.speech_token_size] = -float("inf")
            top_ids = self.sampling_ids(
                logp.squeeze(dim=0),
                out_tokens,
                sampling,
                ignore_eos=True if i < min_len else False,
            ).item()
            if top_ids == self.speech_token_size:
                break
            yield top_ids
            out_tokens.append(top_ids)
            offset += lm_input.size(1)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)


class Qwen2Encoder(paddle.nn.Layer):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, xs: paddle.Tensor, xs_lens: paddle.Tensor):
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T)
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=masks,
            output_hidden_states=True,
            return_dict=True,
        )
        return outs.hidden_states[-1], masks.unsqueeze(1)

    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]
        new_cache = outs.past_key_values
        xs = paddle.cast(xs, dtype = 'float32')
        return xs, new_cache


class Qwen2LM(TransformerLM):
    def __init__(
        self,
        llm_input_size: int,
        llm_output_size: int,
        speech_token_size: int,
        llm: paddle.nn.Layer,
        sampling: Callable,
        length_normalized_loss: bool = True,
        lsm_weight: float = 0.0,
        mix_ratio: List[int] = [5, 15],
    ):
        paddle.nn.Layer.__init__(self)
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size
        self.sos_eos = 0
        self.task_id = 1
        self.fill_token = 2
        self.llm_embedding = paddle.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = paddle.nn.Linear(
            in_features=llm_output_size, out_features=speech_token_size + 3
        )
        # self.llm_decoder.weight = paddle.create_parameter(
        #     shape=self.llm_decoder.weight.shape,
        #     dtype='bfloat16',
        #     default_initializer=paddle.nn.initializer.Assign(self.llm_decoder.weight.astype('bfloat16'))
        # )
        # if self.llm_decoder.bias is not None:
        #     self.llm_decoder.bias = paddle.create_parameter(
        #         shape=self.llm_decoder.bias.shape,
        #         dtype='bfloat16',
        #         default_initializer=paddle.nn.initializer.Assign(self.llm_decoder.bias.astype('bfloat16'))
        #     )
        # self.criterion_ce = LabelSmoothingLoss(
        #     size=speech_token_size + 3,
        #     padding_idx=IGNORE_ID,
        #     smoothing=lsm_weight,
        #     normalize_length=length_normalized_loss,
        # )
        self.speech_embedding = paddle.nn.Embedding(
            speech_token_size + 3, llm_input_size
        )
        self.sampling = sampling
        self.mix_ratio = mix_ratio
        self.stop_token_ids = [(speech_token_size + i) for i in range(3)]
        self.vllm_output_queue = {}

    # def prepare_lm_input_target(
    #     self,
    #     text_token,
    #     text_token_emb,
    #     text_token_len,
    #     speech_token,
    #     speech_token_emb,
    #     speech_token_len,
    # ):
    #     lm_target, lm_input = [], []
    #     text_token = torch.nn.utils.rnn.unpad_sequence(
    #         text_token, text_token_len.cpu(), batch_first=True
    #     )
    #     speech_token = torch.nn.utils.rnn.unpad_sequence(
    #         speech_token, speech_token_len.cpu(), batch_first=True
    #     )
    #     text_token_emb = torch.nn.utils.rnn.unpad_sequence(
    #         text_token_emb, text_token_len.cpu(), batch_first=True
    #     )
    #     speech_token_emb = torch.nn.utils.rnn.unpad_sequence(
    #         speech_token_emb, speech_token_len.cpu(), batch_first=True
    #     )
    #     for i in range(len(text_token)):
    #         if (
    #             random.random() < 0.5
    #             and speech_token_len[i] / text_token_len[i]
    #             > self.mix_ratio[1] / self.mix_ratio[0]
    #         ):
    #             this_lm_target, this_lm_input = [], []
    #             this_lm_target.append(IGNORE_ID)
    #             this_lm_input.append(
    #                 self.llm_embedding.weight[self.sos_eos].reshape(1, -1)
    #             )
    #             for j in range(
    #                 ((text_token_len[i] + 1) / self.mix_ratio[0]).ceil().int().item()
    #             ):
    #                 this_text_token = text_token[i][
    #                     j * self.mix_ratio[0] : (j + 1) * self.mix_ratio[0]
    #                 ].tolist()
    #                 this_speech_token = speech_token[i][
    #                     j * self.mix_ratio[1] : (j + 1) * self.mix_ratio[1]
    #                 ].tolist()
    #                 if len(this_text_token) == self.mix_ratio[0]:
    #                     assert len(this_speech_token) == self.mix_ratio[1]
    #                     this_lm_target += [IGNORE_ID] * (self.mix_ratio[0] - 1)
    #                     this_lm_target += this_speech_token
    #                     this_lm_target.append(self.speech_token_size + 2)
    #                     this_lm_input.append(
    #                         text_token_emb[i][
    #                             j * self.mix_ratio[0] : (j + 1) * self.mix_ratio[0]
    #                         ]
    #                     )
    #                     this_lm_input.append(
    #                         speech_token_emb[i][
    #                             j * self.mix_ratio[1] : (j + 1) * self.mix_ratio[1]
    #                         ]
    #                     )
    #                 else:
    #                     this_lm_target += [-1] * len(this_text_token)
    #                     this_lm_target += speech_token[i][
    #                         j * self.mix_ratio[1] :
    #                     ].tolist()
    #                     this_lm_target.append(self.speech_token_size)
    #                     this_lm_input.append(text_token_emb[i][j * self.mix_ratio[0] :])
    #                     this_lm_input.append(
    #                         self.llm_embedding.weight[self.task_id].reshape(1, -1)
    #                     )
    #                     this_lm_input.append(
    #                         speech_token_emb[i][j * self.mix_ratio[1] :]
    #                     )
    #             this_lm_target, this_lm_input = paddle.tensor(
    #                 this_lm_target
    #             ), paddle.cat(this_lm_input, dim=0)
    #         else:
    #             this_lm_target = paddle.tensor(
    #                 [IGNORE_ID] * (1 + text_token_len[i])
    #                 + speech_token[i].tolist()
    #                 + [self.speech_token_size]
    #             )
    #             this_lm_input = paddle.cat(
    #                 [
    #                     self.llm_embedding.weight[self.sos_eos].reshape(1, -1),
    #                     text_token_emb[i],
    #                     self.llm_embedding.weight[self.task_id].reshape(1, -1),
    #                     speech_token_emb[i],
    #                 ],
    #                 dim=0,
    #             )
    #         lm_target.append(this_lm_target)
    #         lm_input.append(this_lm_input)
    #     lm_input_len = paddle.tensor([i.size(0) for i in lm_input], dtype=paddle.int32)
    #     lm_input = torch.nn.utils.rnn.pad_sequence(
    #         lm_input, batch_first=True, padding_value=IGNORE_ID
    #     )
    #     lm_target = torch.nn.utils.rnn.pad_sequence(
    #         lm_target, batch_first=True, padding_value=IGNORE_ID
    #     )
    #     return lm_target, lm_input, lm_input_len

    @paddle.no_grad()
    def inference(
        self,
        text: paddle.Tensor,
        text_len: paddle.Tensor,
        prompt_text: paddle.Tensor,
        prompt_text_len: paddle.Tensor,
        prompt_speech_token: paddle.Tensor,
        prompt_speech_token_len: paddle.Tensor,
        embedding: paddle.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
        uuid: str = "",
    ) -> Generator[paddle.Tensor, None, None]:
        device = text.place
        text = paddle.cat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.llm.model.qwen2.embed_tokens(text)
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape([1, 1, -1])
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape([1, 1, -1])
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = paddle.zeros(
                1, 0, self.llm_input_size, dtype=text.dtype
            ).to(device)
        text = paddle.cast(text,dtype = 'float32')
        lm_input = paddle.cat(
            [sos_eos_emb, text, task_id_emb, prompt_speech_token_emb], dim=1
        )
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)
        for token in self.inference_wrapper(lm_input, sampling, min_len, max_len, uuid):
            yield token

    @paddle.no_grad()
    def inference_wrapper(self, lm_input, sampling, min_len, max_len, uuid):
        if hasattr(self, "vllm"):
            from vllm import RequestOutput, SamplingParams

            sampling_params = SamplingParams(
                top_k=sampling,
                stop_token_ids=self.stop_token_ids,
                min_tokens=min_len,
                max_tokens=max_len,
            )
            with self.lock:
                self.vllm.add_request(
                    uuid,
                    {
                        "prompt_embeds": lm_input.squeeze(0)
                        .to(paddle.bfloat16)
                        .to(lm_input.place)
                    },
                    sampling_params,
                )
                self.vllm_output_queue[uuid] = queue.Queue()
            out_tokens = []
            while True:
                with self.lock:
                    if self.vllm_output_queue[uuid].empty() is True:
                        request_outputs: List[RequestOutput] = self.vllm.step()
                        for request_output in request_outputs:
                            top_ids = list(request_output.outputs[0].token_ids)[-1]
                            self.vllm_output_queue[request_output.request_id].put(
                                top_ids
                            )
                if self.vllm_output_queue[uuid].empty() is False:
                    top_ids = self.vllm_output_queue[uuid].get()
                    if top_ids in self.stop_token_ids:
                        break
                    yield top_ids
                    out_tokens.append(top_ids)
                    if len(out_tokens) == max_len:
                        break
                time.sleep(0.001)
            with self.lock:
                self.vllm_output_queue.pop(uuid)
        else:
            out_tokens = []
            cache = None
            for i in range(max_len):
                y_pred, cache = self.llm.forward_one_step(
                    lm_input,
                    masks=paddle.tril(
                        paddle.ones(
                            (1, lm_input.shape[1], lm_input.shape[1]),
                        )
                    ).to(paddle.bool),
                    cache=cache,
                )
                logp = F.log_softmax(self.llm_decoder(y_pred[:, -1]), axis = -1)
                top_ids = self.sampling_ids(
                    logp.squeeze(axis=0),
                    out_tokens,
                    sampling,
                    ignore_eos=True if i < min_len else False,
                ).item()
                if top_ids == self.speech_token_size:
                    break
                if top_ids > self.speech_token_size:
                    continue
                yield top_ids
                out_tokens.append(top_ids)
                lm_input = self.speech_embedding.weight[top_ids].reshape([1, 1, -1])

    @paddle.no_grad()
    def inference_bistream(
        self,
        text: Generator,
        prompt_text: paddle.Tensor,
        prompt_text_len: paddle.Tensor,
        prompt_speech_token: paddle.Tensor,
        prompt_speech_token_len: paddle.Tensor,
        embedding: paddle.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
    ) -> Generator[paddle.Tensor, None, None]:
        device = prompt_text.place
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = paddle.zeros(
                1, 0, self.llm_input_size, dtype=prompt_text.dtype
            ).to(device)
        lm_input = paddle.cat([sos_eos_emb], dim=1)
        out_tokens = []
        cache = None
        text_cache = self.llm.model.model.embed_tokens(prompt_text)
        next_fill_index = -1
        for this_text in text:
            text_cache = paddle.cat(
                [text_cache, self.llm.model.model.embed_tokens(this_text)], dim=1
            )
            while prompt_speech_token_emb.size(1) != 0:
                if text_cache.size(1) >= self.mix_ratio[0]:
                    lm_input_text, lm_input_speech = (
                        text_cache[:, : self.mix_ratio[0]],
                        prompt_speech_token_emb[:, : self.mix_ratio[1]],
                    )
                    logging.info(
                        "append {} text token {} speech token".format(
                            lm_input_text.size(1), lm_input_speech.size(1)
                        )
                    )
                    lm_input = paddle.cat(
                        [lm_input, lm_input_text, lm_input_speech], dim=1
                    )
                    text_cache, prompt_speech_token_emb = (
                        text_cache[:, self.mix_ratio[0] :],
                        prompt_speech_token_emb[:, self.mix_ratio[1] :],
                    )
                else:
                    logging.info("not enough text token to decode, wait for more")
                    break
            if prompt_speech_token_emb.size(1) == 0:
                if (
                    len(out_tokens) != 0
                    and out_tokens[-1] == self.speech_token_size + 2
                    or len(out_tokens) == 0
                    and lm_input.size(1) == 1
                ):
                    logging.info("get fill token, need to append more text token")
                    if text_cache.size(1) >= self.mix_ratio[0]:
                        lm_input_text = text_cache[:, : self.mix_ratio[0]]
                        logging.info(
                            "append {} text token".format(lm_input_text.size(1))
                        )
                        if (
                            len(out_tokens) != 0
                            and out_tokens[-1] == self.speech_token_size + 2
                        ):
                            lm_input = lm_input_text
                        else:
                            lm_input = paddle.cat([lm_input, lm_input_text], dim=1)
                        text_cache = text_cache[:, self.mix_ratio[0] :]
                    else:
                        logging.info("not enough text token to decode, wait for more")
                        continue
                while True:
                    seq_len = (
                        lm_input.shape[1]
                        if cache is None
                        else lm_input.shape[1] + cache[0][0].size(2)
                    )
                    y_pred, cache = self.llm.forward_one_step(
                        lm_input,
                        masks=paddle.tril(
                            paddle.ones((1, seq_len, seq_len), device=lm_input.place)
                        ).to(paddle.bool),
                        cache=cache,
                    )
                    logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
                    if next_fill_index != -1 and len(out_tokens) == next_fill_index:
                        top_ids = self.speech_token_size + 2
                        next_fill_index += self.mix_ratio[1] + 1
                    else:
                        top_ids = self.sampling_ids(
                            logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True
                        ).item()
                    if top_ids == self.speech_token_size + 2:
                        next_fill_index = len(out_tokens) + self.mix_ratio[1] + 1
                        logging.info(
                            "fill_token index {} next fill_token index {}".format(
                                len(out_tokens), next_fill_index
                            )
                        )
                    out_tokens.append(top_ids)
                    if top_ids >= self.speech_token_size:
                        if top_ids == self.speech_token_size + 2:
                            break
                        else:
                            raise ValueError("should not get token {}".format(top_ids))
                    yield top_ids
                    lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
        lm_input = paddle.cat([lm_input, text_cache, task_id_emb], dim=1)
        logging.info("no more text token, decode until met eos")
        while True:
            seq_len = (
                lm_input.shape[1]
                if cache is None
                else lm_input.shape[1] + cache[0][0].size(2)
            )
            y_pred, cache = self.llm.forward_one_step(
                lm_input,
                masks=paddle.tril(
                    paddle.ones((1, seq_len, seq_len), device=lm_input.place)
                ).to(paddle.bool),
                cache=cache,
            )
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(
                logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=False
            ).item()
            out_tokens.append(top_ids)
            if top_ids >= self.speech_token_size:
                if top_ids == self.speech_token_size:
                    break
                else:
                    raise ValueError("should not get token {}".format(top_ids))
            yield top_ids
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
