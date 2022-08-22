import numpy as np
import os

from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlespeech.s2t.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ConfigPure
from paddlespeech.s2t.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model


from paddlespeech.s2t.models.wav2vec2.speechbrain.lobes.models.VanillaNN import VanillaNN
from paddlespeech.s2t.modules.ctc import CTCDecoderBase as CTC
from paddlespeech.s2t.utils.ctc_utils import remove_duplicates_and_blank
from yacs.config import CfgNode

class Wav2vec2ASR(nn.Layer):
    def __init__(self, config: dict):
        super().__init__()
        
        wav2vec2_config = Wav2Vec2ConfigPure()
        wav2vec2 = Wav2Vec2Model(wav2vec2_config)

        model_dict = paddle.load(config.wav2vec2_params_path)
        wav2vec2.set_state_dict(model_dict)
        wav2vec2.training = True
        wav2vec2.eval()
        self.normalize_wav = config.normalize_wav
        self.output_norm = config.output_norm
        if config.freeze_wav2vec2:
            for parm in wav2vec2.parameters():
                parm.requires_grad = False
        self.wav2vec2 = wav2vec2
        self.enc = VanillaNN(input_shape=[None,None,wav2vec2_config.hidden_size], activation=nn.LeakyReLU, dnn_blocks=config.dnn_blocks, dnn_neurons=config.dnn_neurons)
        self.ctc = CTC(odim=config.output_dim, enc_n_units=config.dnn_neurons, blank_id=config.blank_id, dropout_rate=config.ctc_dropout_rate, reduction_type="mean")

    def train_batch(self):
        wav, wavs_lens_rate, target, target_lens_rate = self._get_data()
        ctc_loss = self(wav, wavs_lens_rate, target, target_lens_rate)


    def forward(self, wav, wavs_lens_rate, target, target_lens_rate):
        if self.normalize_wav:
            wav = F.layer_norm(wav, wav.shape[1:])
        # Extract wav2vec output
        out = self.wav2vec2(wav)[0]
        np.save("data/out.npy", out.numpy())
        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out, out.shape[1:])
        feats = out

        x = self.enc(feats)
        print ("wavs_lens_rate", wavs_lens_rate)
        print ("x lens", x.shape[1])
        x_lens = (wavs_lens_rate * x.shape[1]).round().astype(paddle.int64)
        target_lens = (target_lens_rate * target.shape[1]).round().astype(paddle.int64)
        
        ctc_loss = self.ctc(x, x_lens, target, target_lens)
        print ("ctc_loss", ctc_loss)
        return ctc_loss

    def decode(self, wav, wavs_lens_rate):
        if self.normalize_wav:
            wav = F.layer_norm(wav, wav.shape[1:])
        # Extract wav2vec output
        out = self.wav2vec2(wav)[0]
        np.save("data/out.npy", out.numpy())
        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out, out.shape[1:])
        feats = out

        x = self.enc(feats)
        x_lens = (wavs_lens_rate * x.shape[1]).round().astype(paddle.int64)

    @classmethod
    def from_config(cls, config):
        model = cls(config)
        return model

    def ctc_greedy_search(
            self, wav, wavs_lens_rate) -> List[List[int]]:
        """ Apply CTC greedy search
        Args:
            speech (paddle.Tensor): (batch, max_len)
            speech_length (paddle.Tensor): (batch, )
        Returns:
            List[List[int]]: best path result
        """
        if self.normalize_wav:
            wav = F.layer_norm(wav, wav.shape[1:])
        # Extract wav2vec output
        out = self.wav2vec2(wav)[0]
        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out, out.shape[1:])
        feats = out

        x = self.enc(feats)
        x_lens = (wavs_lens_rate * x.shape[1]).round().astype(paddle.int64)


        ctc_probs = self.ctc.log_softmax(x)  # (B, maxlen, vocab_size)

        topk_prob, topk_index = ctc_probs.topk(1, axis=2)  # (B, maxlen, 1)
        topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
        pad_mask = make_pad_mask(x_lens)  # (B, maxlen)
        topk_index = topk_index.masked_fill_(pad_mask, self.eos)  # (B, maxlen)

        hyps = [hyp.tolist() for hyp in topk_index]
        hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]
        return hyps

    def _ctc_prefix_beam_search(
           self, wav, wavs_lens_rate) -> Tuple[List[Tuple[int, float]], paddle.Tensor]:
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
        if self.normalize_wav:
            wav = F.layer_norm(wav, wav.shape[1:])
        # Extract wav2vec output
        out = self.wav2vec2(wav)[0]
        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out, out.shape[1:])
        feats = out

        x = self.enc(feats)
        x_lens = (wavs_lens_rate * x.shape[1]).round().astype(paddle.int64)
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

    # @jit.to_static
    def ctc_activation(self, xs: paddle.Tensor) -> paddle.Tensor:
        """ Export interface for c++ call, apply linear transform and log
            softmax before ctc
        Args:
            xs (paddle.Tensor): encoder output, (B, T, D)
        Returns:
            paddle.Tensor: activation before ctc
        """
        return self.ctc.log_softmax(xs)

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
               simulate_streaming: bool=False):
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
      
        else:
            raise ValueError(f"Not support decoding method: {decoding_method}")

        res = [text_feature.defeaturize(hyp) for hyp in hyps]
        res_tokenids = [hyp for hyp in hyps]
        return res, res_tokenids



    def _get_data(self):
        data_dir = "data"
        wavs = np.load(os.path.join(data_dir, "wavs.npy"))
        wavs_lens = np.load(os.path.join(data_dir, "wavs_lens.npy"))
        tokens = np.load(os.path.join(data_dir, "tokens.npy"))
        tokens_lens = np.load(os.path.join(data_dir, "tokens_lens.npy"))
        
        batch = (paddle.to_tensor(wavs), paddle.to_tensor(wavs_lens, dtype='float32'), 
            paddle.to_tensor(tokens, dtype='int32'), paddle.to_tensor(tokens_lens, dtype='float32'))
        return batch


if __name__ == "__main__":
    wav2vec2_asr = Wav2vec2ASR(configs={})
    wav2vec2_asr.train_batch()
    exit(1)

    freeze = True
    config = Wav2Vec2ConfigPure()
    model = Wav2Vec2Model(config)
    model_dict = model.state_dict()
    revise_params_path = "exp/torch_to_paddle_revise.pdparams"
    model_dict_revise = paddle.load(revise_params_path)
    model.set_state_dict(model_dict_revise)
    model.training = True
    model.eval()
    if freeze:
        for parm in model.parameters():
            parm.requires_grad = False
   # get enc()
    enc = VanillaNN(input_shape=[None,None,1024], activation=paddle.nn.LeakyReLU, dnn_blocks=2, dnn_neurons=1024)

    ctc = CTC(odim=30, enc_n_units=1024, blank_id=0, dropout_rate=0.0)

    input_values = np.load("input_values.npy")
    input_values = paddle.to_tensor(input_values)

    feats = model(input_values).last_hidden_state
    x = enc(feats)
    ctc_loss = ctc(enc, target)
