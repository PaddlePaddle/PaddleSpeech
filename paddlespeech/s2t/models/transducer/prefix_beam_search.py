from typing import List
from typing import Tuple

import paddle
import paddle.nn.functional as F

from paddlespeech.s2t.utils.utility import log_add


class Sequence():

    __slots__ = {'hyp', 'score', 'cache'}

    def __init__(
            self,
            hyp: List[paddle.Tensor],
            score,
            cache: List[paddle.Tensor], ):
        self.hyp = hyp
        self.score = score
        self.cache = cache


class PrefixBeamSearch():
    def __init__(self, encoder, predictor, joint, ctc, blank):
        self.encoder = encoder
        self.predictor = predictor
        self.joint = joint
        self.ctc = ctc
        self.blank = blank

    def forward_decoder_one_step(self,
                                 encoder_x: paddle.Tensor,
                                 pre_t: paddle.Tensor,
                                 cache: List[paddle.Tensor]
                                 ) -> Tuple[paddle.Tensor, List[paddle.Tensor]]:
        padding = paddle.zeros([pre_t.shape[0], 1])
        pre_t, new_cache = self.predictor.forward_step(
            pre_t.unsqueeze(-1), padding, cache)
        x = self.joint(encoder_x, pre_t)  # [beam, 1, 1, vocab]
        x = F.log_softmax(x, axis=-1)  #x.log_softmax(dim=-1)
        return x, new_cache

    def prefix_beam_search(self,
                           speech: paddle.Tensor,
                           speech_lengths: paddle.Tensor,
                           decoding_chunk_size: int=-1,
                           beam_size: int=5,
                           num_decoding_left_chunks: int=-1,
                           simulate_streaming: bool=False,
                           ctc_weight: float=0.3,
                           transducer_weight: float=0.7):
        """prefix beam search
           also see wenet.transducer.transducer.beam_search
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        #device = speech.device
        batch_size = speech.shape[0]
        assert batch_size == 1

        # 1. Encoder
        encoder_out, _ = self.encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.shape[1]

        ctc_probs = self.ctc.log_softmax(encoder_out).squeeze(0)
        beam_init: List[Sequence] = []

        # 2. init beam using Sequence to save beam unit
        cache = self.predictor.init_state(1, method="zero")
        beam_init.append(Sequence(hyp=[self.blank], score=0.0, cache=cache))
        # 3. start decoding (notice: we use breathwise first searching)
        # !!!! In this decoding method: one frame do not output multi units. !!!!
        # !!!!    Experiments show that this strategy has little impact      !!!!
        for i in range(maxlen):
            # 3.1 building input
            # decoder taking the last token to predict the next token
            input_hyp = [s.hyp[-1] for s in beam_init]
            input_hyp_tensor = paddle.to_tensor(
                input_hyp,
                dtype=paddle.int, )
            # building statement from beam
            cache_batch = self.predictor.cache_to_batch(
                [s.cache for s in beam_init])
            # build score tensor to do torch.add() function
            scores = paddle.to_tensor([s.score for s in beam_init])

            # 3.2 forward decoder
            logp, new_cache = self.forward_decoder_one_step(
                encoder_out[:, i, :].unsqueeze(1),
                input_hyp_tensor,
                cache_batch, )  # logp: (N, 1, 1, vocab_size)
            logp = logp.squeeze(1).squeeze(1)  # logp: (N, vocab_size)
            new_cache = self.predictor.batch_to_cache(new_cache)

            # 3.3 shallow fusion for transducer score
            #     and ctc score where we can also add the LM score
            logp = paddle.log(
                paddle.add(transducer_weight * paddle.exp(logp), ctc_weight *
                           paddle.exp(ctc_probs[i].unsqueeze(0))))

            # 3.4 first beam prune
            top_k_logp, top_k_index = logp.topk(beam_size)  # (N, N)
            scores = paddle.add(scores.unsqueeze(1), top_k_logp)

            # 3.5 generate new beam (N*N)
            beam_A = []
            for j in range(len(beam_init)):
                # update seq
                base_seq = beam_init[j]
                for t in range(beam_size):
                    # blank: only update the score
                    if top_k_index[j, t] == self.blank:
                        new_seq = Sequence(
                            hyp=base_seq.hyp.copy(),
                            score=scores[j, t].item(),
                            cache=base_seq.cache)

                        beam_A.append(new_seq)
                    # other unit: update hyp score statement and last
                    else:
                        hyp_new = base_seq.hyp.copy()
                        hyp_new.append(top_k_index[j, t].item())
                        #print ("hyp_new:{}, score_size:{}, j:{}, t:{}, new_cache:{}".format(hyp_new,scores.shape, j,t, len(new_cache)))
                        new_seq = Sequence(
                            hyp=hyp_new,
                            score=scores[j, t].item(),
                            cache=new_cache[j])
                        beam_A.append(new_seq)

            # 3.6 prefix fusion
            fusion_A = [beam_A[0]]
            for j in range(1, len(beam_A)):
                s1 = beam_A[j]
                if_do_append = True
                for t in range(len(fusion_A)):
                    # notice: A_ can not fusion with A
                    if s1.hyp == fusion_A[t].hyp:
                        fusion_A[t].score = log_add(
                            [fusion_A[t].score, s1.score])
                        if_do_append = False
                        break
                if if_do_append:
                    fusion_A.append(s1)

            # 4. second pruned
            fusion_A.sort(key=lambda x: x.score, reverse=True)
            beam_init = fusion_A[:beam_size]

        return beam_init, encoder_out
