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
import paddle
from paddle import nn

from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()

__all__ = ['CRF']


class CRF(nn.Layer):
    """
    Linear-chain Conditional Random Field (CRF).
    
    Args:
        nb_labels (int): number of labels in your tagset, including special symbols.
        bos_tag_id (int): integer representing the beginning of sentence symbol in
            your tagset.
        eos_tag_id (int): integer representing the end of sentence symbol in your tagset.
        pad_tag_id (int, optional): integer representing the pad symbol in your tagset.
            If None, the model will treat the PAD as a normal tag. Otherwise, the model
            will apply constraints for PAD transitions.
        batch_first (bool): Whether the first dimension represents the batch dimension.
    """

    def __init__(self,
                 nb_labels: int,
                 bos_tag_id: int,
                 eos_tag_id: int,
                 pad_tag_id: int=None,
                 batch_first: bool=True):
        super().__init__()

        self.nb_labels = nb_labels
        self.BOS_TAG_ID = bos_tag_id
        self.EOS_TAG_ID = eos_tag_id
        self.PAD_TAG_ID = pad_tag_id
        self.batch_first = batch_first

        # initialize transitions from a random uniform distribution between -0.1 and 0.1
        self.transitions = self.create_parameter(
            [self.nb_labels, self.nb_labels],
            default_initializer=nn.initializer.Uniform(-0.1, 0.1))
        self.init_weights()

    def init_weights(self):
        # enforce contraints (rows=from, columns=to) with a big negative number
        # so exp(-10000) will tend to zero

        # no transitions allowed to the beginning of sentence
        self.transitions[:, self.BOS_TAG_ID] = -10000.0
        # no transition alloed from the end of sentence
        self.transitions[self.EOS_TAG_ID, :] = -10000.0

        if self.PAD_TAG_ID is not None:
            # no transitions from padding
            self.transitions[self.PAD_TAG_ID, :] = -10000.0
            # no transitions to padding
            self.transitions[:, self.PAD_TAG_ID] = -10000.0
            # except if the end of sentence is reached
            # or we are already in a pad position
            self.transitions[self.PAD_TAG_ID, self.EOS_TAG_ID] = 0.0
            self.transitions[self.PAD_TAG_ID, self.PAD_TAG_ID] = 0.0

    def forward(self,
                emissions: paddle.Tensor,
                tags: paddle.Tensor,
                mask: paddle.Tensor=None) -> paddle.Tensor:
        """Compute the negative log-likelihood. See `log_likelihood` method."""
        nll = -self.log_likelihood(emissions, tags, mask=mask)
        return nll

    def log_likelihood(self, emissions, tags, mask=None):
        """Compute the probability of a sequence of tags given a sequence of
        emissions scores.

        Args:
            emissions (paddle.Tensor): Sequence of emissions for each label.
                Shape of (batch_size, seq_len, nb_labels) if batch_first is True,
                (seq_len, batch_size, nb_labels) otherwise.
            tags (paddle.LongTensor): Sequence of labels.
                Shape of (batch_size, seq_len) if batch_first is True,
                (seq_len, batch_size) otherwise.
            mask (paddle.FloatTensor, optional): Tensor representing valid positions.
                If None, all positions are considered valid.
                Shape of (batch_size, seq_len) if batch_first is True,
                (seq_len, batch_size) otherwise.

        Returns:
            paddle.Tensor: sum of the log-likelihoods for each sequence in the batch.
                Shape of ()
        """
        # fix tensors order by setting batch as the first dimension
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)

        if mask is None:
            mask = paddle.ones(emissions.shape[:2], dtype=paddle.float)

        scores = self._compute_scores(emissions, tags, mask=mask)
        partition = self._compute_log_partition(emissions, mask=mask)
        return paddle.sum(scores - partition)

    def decode(self, emissions, mask=None):
        """Find the most probable sequence of labels given the emissions using
        the Viterbi algorithm.

        Args:
            emissions (paddle.Tensor): Sequence of emissions for each label.
                Shape (batch_size, seq_len, nb_labels) if batch_first is True,
                (seq_len, batch_size, nb_labels) otherwise.
            mask (paddle.FloatTensor, optional): Tensor representing valid positions.
                If None, all positions are considered valid.
                Shape (batch_size, seq_len) if batch_first is True,
                (seq_len, batch_size) otherwise.

        Returns:
            paddle.Tensor: the viterbi score for the for each batch.
                Shape of (batch_size,)
            list of lists: the best viterbi sequence of labels for each batch. [B, T]
        """
        # fix tensors order by setting batch as the first dimension
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)

        if mask is None:
            mask = paddle.ones(emissions.shape[:2], dtype=paddle.float)

        scores, sequences = self._viterbi_decode(emissions, mask)
        return scores, sequences

    def _compute_scores(self, emissions, tags, mask):
        """Compute the scores for a given batch of emissions with their tags.

        Args:
            emissions (paddle.Tensor): (batch_size, seq_len, nb_labels)
            tags (Paddle.LongTensor): (batch_size, seq_len)
            mask (Paddle.FloatTensor): (batch_size, seq_len)

        Returns:
            paddle.Tensor: Scores for each batch.
                Shape of (batch_size,)
        """
        batch_size, seq_length = tags.shape
        scores = paddle.zeros([batch_size])

        # save first and last tags to be used later
        first_tags = tags[:, 0]
        last_valid_idx = mask.int().sum(1) - 1

        # TODO(Hui Zhang): not support fancy index. 
        # last_tags = tags.gather(last_valid_idx.unsqueeze(1), axis=1).squeeze()
        batch_idx = paddle.arange(batch_size, dtype=last_valid_idx.dtype)
        gather_last_valid_idx = paddle.stack(
            [batch_idx, last_valid_idx], axis=-1)
        last_tags = tags.gather_nd(gather_last_valid_idx)

        # add the transition from BOS to the first tags for each batch
        # t_scores = self.transitions[self.BOS_TAG_ID, first_tags]
        t_scores = self.transitions[self.BOS_TAG_ID].gather(first_tags)

        # add the [unary] emission scores for the first tags for each batch
        # for all batches, the first word, see the correspondent emissions
        # for the first tags (which is a list of ids):
        # emissions[:, 0, [tag_1, tag_2, ..., tag_nblabels]]
        # e_scores = emissions[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze()
        gather_first_tags_idx = paddle.stack([batch_idx, first_tags], axis=-1)
        e_scores = emissions[:, 0].gather_nd(gather_first_tags_idx)

        # the scores for a word is just the sum of both scores
        scores += e_scores + t_scores

        # now lets do this for each remaining word
        for i in range(1, seq_length):

            # we could: iterate over batches, check if we reached a mask symbol
            # and stop the iteration, but vecotrizing is faster due to gpu,
            # so instead we perform an element-wise multiplication
            is_valid = mask[:, i]

            previous_tags = tags[:, i - 1]
            current_tags = tags[:, i]

            # calculate emission and transition scores as we did before
            # e_scores = emissions[:, i].gather(1, current_tags.unsqueeze(1)).squeeze()
            gather_current_tags_idx = paddle.stack(
                [batch_idx, current_tags], axis=-1)
            e_scores = emissions[:, i].gather_nd(gather_current_tags_idx)
            # t_scores = self.transitions[previous_tags, current_tags]
            gather_transitions_idx = paddle.stack(
                [previous_tags, current_tags], axis=-1)
            t_scores = self.transitions.gather_nd(gather_transitions_idx)

            # apply the mask
            e_scores = e_scores * is_valid
            t_scores = t_scores * is_valid

            scores += e_scores + t_scores

        # add the transition from the end tag to the EOS tag for each batch
        # scores += self.transitions[last_tags, self.EOS_TAG_ID]
        scores += self.transitions.gather(last_tags)[:, self.EOS_TAG_ID]

        return scores

    def _compute_log_partition(self, emissions, mask):
        """Compute the partition function in log-space using the forward-algorithm.

        Args:
            emissions (paddle.Tensor): (batch_size, seq_len, nb_labels)
            mask (Paddle.FloatTensor): (batch_size, seq_len)

        Returns:
            paddle.Tensor: the partition scores for each batch.
                Shape of (batch_size,)
        """
        batch_size, seq_length, nb_labels = emissions.shape

        # in the first iteration, BOS will have all the scores
        alphas = self.transitions[self.BOS_TAG_ID, :].unsqueeze(
            0) + emissions[:, 0]

        for i in range(1, seq_length):
            # (bs, nb_labels) -> (bs, 1, nb_labels)
            e_scores = emissions[:, i].unsqueeze(1)

            # (nb_labels, nb_labels) -> (bs, nb_labels, nb_labels)
            t_scores = self.transitions.unsqueeze(0)

            # (bs, nb_labels)  -> (bs, nb_labels, 1)
            a_scores = alphas.unsqueeze(2)

            scores = e_scores + t_scores + a_scores
            new_alphas = paddle.logsumexp(scores, axis=1)

            # set alphas if the mask is valid, otherwise keep the current values
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * new_alphas + (1 - is_valid) * alphas

        # add the scores for the final transition
        last_transition = self.transitions[:, self.EOS_TAG_ID]
        end_scores = alphas + last_transition.unsqueeze(0)

        # return a *log* of sums of exps
        return paddle.logsumexp(end_scores, axis=1)

    def _viterbi_decode(self, emissions, mask):
        """Compute the viterbi algorithm to find the most probable sequence of labels
        given a sequence of emissions.

        Args:
            emissions (paddle.Tensor): (batch_size, seq_len, nb_labels)
            mask (Paddle.FloatTensor): (batch_size, seq_len)

        Returns:
            paddle.Tensor: the viterbi score for the for each batch.
                Shape of (batch_size,)
            list of lists of ints: the best viterbi sequence of labels for each batch
        """
        batch_size, seq_length, nb_labels = emissions.shape

        # in the first iteration, BOS will have all the scores and then, the max
        alphas = self.transitions[self.BOS_TAG_ID, :].unsqueeze(
            0) + emissions[:, 0]

        backpointers = []

        for i in range(1, seq_length):
            # (bs, nb_labels) -> (bs, 1, nb_labels)
            e_scores = emissions[:, i].unsqueeze(1)

            # (nb_labels, nb_labels) -> (bs, nb_labels, nb_labels)
            t_scores = self.transitions.unsqueeze(0)

            # (bs, nb_labels)  -> (bs, nb_labels, 1)
            a_scores = alphas.unsqueeze(2)

            # combine current scores with previous alphas
            scores = e_scores + t_scores + a_scores

            # so far is exactly like the forward algorithm,
            # but now, instead of calculating the logsumexp,
            # we will find the highest score and the tag associated with it
            # max_scores, max_score_tags = paddle.max(scores, axis=1)
            max_scores = paddle.max(scores, axis=1)
            max_score_tags = paddle.argmax(scores, axis=1)

            # set alphas if the mask is valid, otherwise keep the current values
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * max_scores + (1 - is_valid) * alphas

            # add the max_score_tags for our list of backpointers
            # max_scores has shape (batch_size, nb_labels) so we transpose it to
            # be compatible with our previous loopy version of viterbi
            backpointers.append(max_score_tags.t())

        # add the scores for the final transition
        last_transition = self.transitions[:, self.EOS_TAG_ID]
        end_scores = alphas + last_transition.unsqueeze(0)

        # get the final most probable score and the final most probable tag
        # max_final_scores, max_final_tags = paddle.max(end_scores, axis=1)
        max_final_scores = paddle.max(end_scores, axis=1)
        max_final_tags = paddle.argmax(end_scores, axis=1)

        # find the best sequence of labels for each sample in the batch
        best_sequences = []
        emission_lengths = mask.int().sum(axis=1)
        for i in range(batch_size):

            # recover the original sentence length for the i-th sample in the batch
            sample_length = emission_lengths[i].item()

            # recover the max tag for the last timestep
            sample_final_tag = max_final_tags[i].item()

            # limit the backpointers until the last but one
            # since the last corresponds to the sample_final_tag
            sample_backpointers = backpointers[:sample_length - 1]

            # follow the backpointers to build the sequence of labels
            sample_path = self._find_best_path(i, sample_final_tag,
                                               sample_backpointers)

            # add this path to the list of best sequences
            best_sequences.append(sample_path)

        return max_final_scores, best_sequences

    def _find_best_path(self, sample_id, best_tag, backpointers):
        """Auxiliary function to find the best path sequence for a specific sample.

            Args:
                sample_id (int): sample index in the range [0, batch_size)
                best_tag (int): tag which maximizes the final score
                backpointers (list of lists of tensors): list of pointers with
                shape (seq_len_i-1, nb_labels, batch_size) where seq_len_i
                represents the length of the ith sample in the batch

            Returns:
                list of ints: a list of tag indexes representing the bast path
        """
        # add the final best_tag to our best path
        best_path = [best_tag]

        # traverse the backpointers in backwards
        for backpointers_t in reversed(backpointers):

            # recover the best_tag at this timestep
            best_tag = backpointers_t[best_tag][sample_id].item()

            # append to the beginning of the list so we don't need to reverse it later
            best_path.insert(0, best_tag)

        return best_path
