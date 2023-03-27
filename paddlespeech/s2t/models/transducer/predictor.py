from typing import List
from typing import Optional
from typing import Tuple

import paddle
from matplotlib.pyplot import axis
from paddle import nn
from typeguard import check_argument_types

from paddlespeech.s2t.utils.rnnt_utils import get_activation
from paddlespeech.s2t.utils.rnnt_utils import get_rnn


def ApplyPadding(input, padding, pad_value) -> paddle.Tensor:
    """
    Args:
        input:   [bs, max_time_step, dim]
        padding: [bs, max_time_step]
    """
    return padding * pad_value + input * (1 - padding)


class PredictorBase(paddle.nn.Layer):

    # NOTE(Mddct): We can use ABC abstract here, but
    # keep this class simple enough for now
    def __init__(self) -> None:
        super().__init__()

    def init_state(self,
                   batch_size: int,
                   device: paddle.device,
                   method: str="zero") -> List[paddle.Tensor]:
        _, _, _ = batch_size, method, device
        raise NotImplementedError("this is a base precictor")

    def batch_to_cache(self,
                       cache: List[paddle.Tensor]) -> List[List[paddle.Tensor]]:
        _ = cache
        raise NotImplementedError("this is a base precictor")

    def cache_to_batch(self,
                       cache: List[List[paddle.Tensor]]) -> List[paddle.Tensor]:
        _ = cache
        raise NotImplementedError("this is a base precictor")

    def forward(
            self,
            input: paddle.Tensor,
            cache: Optional[List[paddle.Tensor]]=None, ):
        _, _, = input, cache
        raise NotImplementedError("this is a base precictor")

    def forward_step(self,
                     input: paddle.Tensor,
                     padding: paddle.Tensor,
                     cache: List[paddle.Tensor]
                     ) -> Tuple[paddle.Tensor, List[paddle.Tensor]]:
        _, _, _, = input, padding, cache
        raise NotImplementedError("this is a base precictor")


class RNNPredictor(PredictorBase):
    def __init__(self,
                 voca_size: int,
                 embed_size: int,
                 output_size: int,
                 embed_dropout: float,
                 hidden_size: int,
                 num_layers: int,
                 bias: bool=True,
                 rnn_type: str="lstm",
                 dropout: float=0.1) -> None:
        assert check_argument_types()
        super().__init__()
        self.n_layers = num_layers
        self.hidden_size = hidden_size
        # disable rnn base out projection
        self.embed = nn.Embedding(voca_size, embed_size)
        self.dropout = nn.Dropout(embed_dropout)
        # NOTE(Mddct): rnn base from torch not support layer norm
        # will add layer norm and prune value in cell and layer
        # ref: https://github.com/Mddct/neural-lm/blob/main/models/gru_cell.py
        self.rnn = get_rnn(rnn_type=rnn_type)(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout)
        self.projection = nn.Linear(hidden_size, output_size)

    def forward(
            self,
            input: paddle.Tensor,
            cache: Optional[List[paddle.Tensor]]=None, ) -> paddle.Tensor:
        """
        Args:
            input (torch.Tensor): [batch, max_time).
            padding (torch.Tensor): [batch, max_time]
            cache : rnn predictor cache[0] == state_m
                    cache[1] == state_c
        Returns:
            output: [batch, max_time, output_size]
        """

        # NOTE(Mddct): we don't use pack input format
        embed = self.embed(input)  # [batch, max_time, emb_size]
        embed = self.dropout(embed)
        states: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None
        if cache is None:
            state = self.init_state(batch_size=input.shape[0])
            states = (state[0], state[1])
        else:
            assert len(cache) == 2
            states = (cache[0], cache[1])
        out, (m, c) = self.rnn(embed, states)
        out = self.projection(out)
        # NOTE(Mddct): Although we don't use staate in transducer
        # training forward, we need make it right for padding value
        # so we create forward_step for infering, forward for training
        _, _ = m, c
        return out

    def batch_to_cache(self,
                       cache: List[paddle.Tensor]) -> List[List[paddle.Tensor]]:
        """
        Args:
           cache: [state_m, state_c]
               state_ms: [1*n_layers, bs, ...]
               state_cs: [1*n_layers, bs, ...]
        Returns:
           new_cache: [[state_m_1, state_c_1], [state_m_2, state_c_2]...]
        """
        assert len(cache) == 2
        state_ms = cache[0]
        state_cs = cache[1]

        assert state_ms.shape[1] == state_cs.shape[1]
        # print ("state_cs:",state_cs.shape,state_ms.shape)
        # print (len(paddle.split(state_ms, 1, axis=1)))

        new_cache: List[List[paddle.Tensor]] = []
        # for state_m, state_c in zip(paddle.split(state_ms, 1, axis=1),
        #                             paddle.split(state_cs, 1, axis=1)):
        for state_m, state_c in zip(
                paddle.split(state_ms, state_ms.shape[1], axis=1),
                paddle.split(state_cs, state_cs.shape[1], axis=1)):
            new_cache.append([state_m, state_c])
            # print ("222", len(new_cache))
        return new_cache

    def cache_to_batch(self,
                       cache: List[List[paddle.Tensor]]) -> List[paddle.Tensor]:
        """
        Args:
            cache : [[state_m_1, state_c_1], [state_m_1, state_c_1]...]

        Returns:
            new_caceh: [state_ms, state_cs],
                state_ms: [1*n_layers, bs, ...]
                state_cs: [1*n_layers, bs, ...]
        """
        state_ms = paddle.concat([states[0] for states in cache], axis=1)
        state_cs = paddle.concat([states[1] for states in cache], axis=1)
        return [state_ms, state_cs]

    def init_state(
            self,
            batch_size: int,
            method: str="zero", ) -> List[paddle.Tensor]:
        assert batch_size > 0
        # TODO(Mddct): xavier init method
        _ = method
        return [
            paddle.zeros([1 * self.n_layers, batch_size, self.hidden_size]),
            paddle.zeros([1 * self.n_layers, batch_size, self.hidden_size])
        ]

    def forward_step(self,
                     input: paddle.Tensor,
                     padding: paddle.Tensor,
                     cache: List[paddle.Tensor]
                     ) -> Tuple[paddle.Tensor, List[paddle.Tensor]]:
        """
        Args:
            input (torch.Tensor): [batch_size, time_step=1]
            padding (torch.Tensor): [batch_size,1], 1 is padding value
            cache : rnn predictor cache[0] == state_m
                    cache[1] == state_c
        """
        assert len(cache) == 2
        state_m, state_c = cache[0], cache[1]
        embed = self.embed(input)  # [batch, 1, emb_size]
        embed = self.dropout(embed)
        out, (m, c) = self.rnn(embed, (state_m, state_c))

        out = self.projection(out)
        m = ApplyPadding(m, padding, state_m)
        c = ApplyPadding(c, padding, state_c)
        return (out, [m, c])


class EmbeddingPredictor(PredictorBase):
    """Embedding predictor

    Described in:
    https://arxiv.org/pdf/2109.07513.pdf

    embed-> proj -> layer norm -> swish
    """

    def __init__(self,
                 voca_size: int,
                 embed_size: int,
                 embed_dropout: float,
                 n_head: int,
                 history_size: int=2,
                 activation: str="silu",
                 bias: bool=False,
                 layer_norm_epsilon: float=1e-5) -> None:

        assert check_argument_types()
        super().__init__()
        # multi head
        self.num_heads = n_head
        self.embed_size = embed_size
        self.context_size = history_size + 1
        self.pos_embed = paddle.nn.Linear(
            embed_size * self.context_size, self.num_heads, bias=bias)
        self.embed = nn.Embedding(voca_size, self.embed_size)
        self.embed_dropout = nn.Dropout(p=embed_dropout)
        self.ffn = nn.Linear(self.embed_size, self.embed_size)
        self.norm = nn.LayerNorm(self.embed_size, eps=layer_norm_epsilon)
        self.activatoin = get_activation(activation)

    def init_state(self,
                   batch_size: int,
                   device: paddle.device,
                   method: str="zero") -> List[paddle.Tensor]:
        assert batch_size > 0
        _ = method
        return [
            paddle.zeros([batch_size, self.context_size - 1, self.embed_size]),
        ]

    def batch_to_cache(self,
                       cache: List[paddle.Tensor]) -> List[List[paddle.Tensor]]:
        """
        Args:
            cache : [history]
                history: [bs, ...]
        Returns:
            new_ache : [[history_1], [history_2], [history_3]...]
        """
        assert len(cache) == 1
        cache_0 = cache[0]
        history: List[List[paddle.Tensor]] = []
        for h in paddle.split(cache_0, 1, axis=0):
            history.append([h])
        return history

    def cache_to_batch(self,
                       cache: List[List[paddle.Tensor]]) -> List[paddle.Tensor]:
        """
        Args:
            cache : [[history_1], [history_2], [history3]...]

        Returns:
            new_caceh: [history],
                history: [bs, ...]
        """
        history = paddle.concat([h[0] for h in cache], axis=0)
        return [history]

    def forward(self,
                input: paddle.Tensor,
                cache: Optional[List[paddle.Tensor]]=None):
        """ forward for training
        """
        input = self.embed(input)  # [bs, seq_len, embed]
        input = self.embed_dropout(input)
        if cache is None:
            zeros = self.init_state(input.shape[0], device=input.device)[0]
        else:
            assert len(cache) == 1
            zeros = cache[0]

        input = paddle.concat(
            (zeros, input), axis=1)  # [bs, context_size-1 + seq_len, embed]

        input = input.unfold(1, self.context_size, 1).permute(
            0, 1, 3, 2)  # [bs, seq_len, context_size, embed]
        # multi head pos: [n_head, embed, context_size]
        multi_head_pos = self.pos_embed.weight.view(
            self.num_heads, self.embed_size, self.context_size)

        # broadcast dot attenton
        input_expand = input.unsqueeze(
            2)  # [bs, seq_len, 1, context_size, embed]
        multi_head_pos = multi_head_pos.permute(
            0, 2, 1)  # [num_heads, context_size, embed]

        # [bs, seq_len, num_heads, context_size, embed]
        weight = input_expand * multi_head_pos
        weight = weight.sum(
            axis=-1, keepdim=False).unsqueeze(
                3)  # [bs, seq_len, num_heads, 1, context_size]
        output = weight.matmul(input_expand).squeeze(
            axis=3)  # [bs, seq_len, num_heads, embed]
        output = output.sum(axis=2)  # [bs, seq_len, embed]
        output = output / (self.num_heads * self.context_size)

        output = self.ffn(output)
        output = self.norm(output)
        output = self.activatoin(output)
        return output

    def forward_step(
            self,
            input: paddle.Tensor,
            padding: paddle.Tensor,
            cache: List[paddle.Tensor],
    ) -> Tuple[paddle.Tensor, List[paddle.Tensor]]:
        """ forward step for inference
        Args:
            input (torch.Tensor): [batch_size, time_step=1]
            padding (torch.Tensor): [batch_size,1], 1 is padding value
            cache: for embedding predictor, cache[0] == history
        """
        assert input.shape[1] == 1
        assert len(cache) == 1
        history = cache[0]
        assert history.shape[1] == self.context_size - 1
        input = self.embed(input)  # [bs, 1, embed]
        input = self.embed_dropout(input)
        context_input = paddle.concat((history, input), axis=1)
        input_expand = context_input.unsqueeze(1).unsqueeze(
            2)  # [bs, 1, 1, context_size, embed]

        # multi head pos: [n_head, embed, context_size]
        multi_head_pos = self.pos_embed.weight.view(
            self.num_heads, self.embed_size, self.context_size)

        multi_head_pos = multi_head_pos.permute(
            0, 2, 1)  # [num_heads, context_size, embed]
        # [bs, 1, num_heads, context_size, embed]
        weight = input_expand * multi_head_pos
        weight = weight.sum(
            axis=-1,
            keepdim=False).unsqueeze(3)  # [bs, 1, num_heads, 1, context_size]
        output = weight.matmul(input_expand).squeeze(
            axis=3)  # [bs, 1, num_heads, embed]
        output = output.sum(axis=2)  # [bs, 1, embed]
        output = output / (self.num_heads * self.context_size)

        output = self.ffn(output)
        output = self.norm(output)
        output = self.activatoin(output)
        new_cache = context_input[:, 1:, :]
        # TODO(Mddct): we need padding new_cache in future
        # new_cache = ApplyPadding(history, padding, new_cache)
        return (output, [new_cache])


class ConvPredictor(PredictorBase):
    def __init__(self,
                 voca_size: int,
                 embed_size: int,
                 embed_dropout: float,
                 history_size: int=2,
                 activation: str="relu",
                 bias: bool=False,
                 layer_norm_epsilon: float=1e-5) -> None:
        assert check_argument_types()
        super().__init__()

        assert history_size >= 0
        self.embed_size = embed_size
        self.context_size = history_size + 1
        self.embed = nn.Embedding(voca_size, self.embed_size)
        self.embed_dropout = nn.Dropout(p=embed_dropout)
        self.conv = nn.Conv1d(
            in_channels=embed_size,
            out_channels=embed_size,
            kernel_size=self.context_size,
            padding=0,
            groups=embed_size,
            bias=bias)
        self.norm = nn.LayerNorm(embed_size, eps=layer_norm_epsilon)
        self.activatoin = get_activation(activation)

    def init_state(self,
                   batch_size: int,
                   device: paddle.device,
                   method: str="zero") -> List[paddle.Tensor]:
        assert batch_size > 0
        assert method == "zero"
        return [
            paddle.zeros([batch_size, self.context_size - 1, self.embed_size])
        ]

    def cache_to_batch(self,
                       cache: List[List[paddle.Tensor]]) -> List[paddle.Tensor]:
        """
        Args:
            cache : [[history_1], [history_2], [history3]...]

        Returns:
            new_caceh: [history],
                history: [bs, ...]
        """
        history = paddle.concat([h[0] for h in cache], axis=0)
        return [history]

    def batch_to_cache(self,
                       cache: List[paddle.Tensor]) -> List[List[paddle.Tensor]]:
        """
        Args:
            cache : [history]
                history: [bs, ...]
        Returns:
            new_ache : [[history_1], [history_2], [history_3]...]
        """
        assert len(cache) == 1
        print(cache[0], cache)
        cache_0 = cache[0]
        history: List[List[paddle.Tensor]] = []
        for h in paddle.split(cache_0, 1, axis=0):
            history.append([h])
        return history

    def forward(self,
                input: paddle.Tensor,
                cache: Optional[List[paddle.Tensor]]=None):
        """ forward for training
        """
        input = self.embed(input)  # [bs, seq_len, embed]
        input = self.embed_dropout(input)
        if cache is None:
            zeros = self.init_state(input.shape[0], device=input.device)[0]
        else:
            assert len(cache) == 1
            zeros = cache[0]

        input = paddle.concat(
            (zeros, input), axis=1)  # [bs, context_size-1 + seq_len, embed]
        input = input.permute(0, 2, 1)
        out = self.conv(input).permute(0, 2, 1)
        out = self.activatoin(self.norm(out))
        return out

    def forward_step(self,
                     input: paddle.Tensor,
                     padding: paddle.Tensor,
                     cache: List[paddle.Tensor]
                     ) -> Tuple[paddle.Tensor, List[paddle.Tensor]]:
        """ forward step for inference
        Args:
            input (torch.Tensor): [batch_size, time_step=1]
            padding (torch.Tensor): [batch_size,1], 1 is padding value
            cache: for embedding predictor, cache[0] == history
        """
        assert input.shape[1] == 1
        assert len(cache) == 1
        history = cache[0]
        assert history.shape[1] == self.context_size - 1
        input = self.embed(input)  # [bs, 1, embed]
        input = self.embed_dropout(input)
        context_input = paddle.concat((history, input), axis=1)
        input = context_input.permute(0, 2, 1)
        out = self.conv(input).permute(0, 2, 1)
        out = self.activatoin(self.norm(out))

        new_cache = context_input[:, 1:, :]
        # TODO(Mddct): apply padding in future
        return (out, [new_cache])
