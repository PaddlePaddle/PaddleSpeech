"""Length bonus module."""
from typing import Any
from typing import List
from typing import Tuple

import paddle

from .scorer_interface import BatchScorerInterface


class LengthBonus(BatchScorerInterface):
    """Length bonus in beam search."""

    def __init__(self, n_vocab: int):
        """Initialize class.

        Args:
            n_vocab (int): The number of tokens in vocabulary for beam search

        """
        self.n = n_vocab

    def score(self, y, state, x):
        """Score new token.

        Args:
            y (paddle.Tensor): 1D paddle.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (paddle.Tensor): 2D encoder feature that generates ys.

        Returns:
            tuple[paddle.Tensor, Any]: Tuple of
                paddle.float32 scores for next token (n_vocab)
                and None

        """
        return paddle.to_tensor([1.0], place=x.place, dtype=x.dtype).expand(self.n), None

    def batch_score(
        self, ys: paddle.Tensor, states: List[Any], xs: paddle.Tensor
    ) -> Tuple[paddle.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (paddle.Tensor): paddle.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (paddle.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[paddle.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        return (
            paddle.to_tensor([1.0], place=xs.place, dtype=xs.dtype).expand(
                ys.shape[0], self.n
            ),
            None,
        )
