"""
Alignment code

Authors
 * Elena Rastorgueva 2020
 * Loren Lugosch 2020
"""
import paddle
import random
from speechbrain.utils.checkpoints import register_checkpoint_hooks
from speechbrain.utils.checkpoints import mark_as_saver
from speechbrain.utils.checkpoints import mark_as_loader
from speechbrain.utils.data_utils import undo_padding


@register_checkpoint_hooks
class HMMAligner(paddle.nn.Layer):
    """This class calculates Viterbi alignments in the forward method.

    It also records alignments and creates batches of them for use
    in Viterbi training.

    Arguments
    ---------
    states_per_phoneme : int
        Number of hidden states to use per phoneme.
    output_folder : str
        It is the folder that the alignments will be stored in when
        saved to disk. Not yet implemented.
    neg_inf : float
        The float used to represent a negative infinite log probability.
        Using `-float("Inf")` tends to give numerical instability.
        A number more negative than -1e5 also sometimes gave errors when
        the `genbmm` library was used (currently not in use). (default: -1e5)
    batch_reduction : string
        One of "none", "sum" or "mean".
        What kind of batch-level reduction to apply to the loss calculated
        in the forward method.
    input_len_norm : bool
        Whether to normalize the loss in the forward method by the length of
        the inputs.
    target_len_norm : bool
        Whether to normalize the loss in the forward method by the length of
        the targets.
    lexicon_path : string
        The location of the lexicon.

    Example
    -------
    >>> log_posteriors = torch.tensor([[[ -1., -10., -10.],
    ...                                 [-10.,  -1., -10.],
    ...                                 [-10., -10.,  -1.]],
    ...
    ...                                [[ -1., -10., -10.],
    ...                                 [-10.,  -1., -10.],
    ...                                 [-10., -10., -10.]]])
    >>> lens = torch.tensor([1., 0.66])
    >>> phns = torch.tensor([[0, 1, 2],
    ...                      [0, 1, 0]])
    >>> phn_lens = torch.tensor([1., 0.66])
    >>> aligner = HMMAligner()
    >>> forward_scores = aligner(
    ...        log_posteriors, lens, phns, phn_lens, 'forward'
    ... )
    >>> forward_scores.shape
    torch.Size([2])
    >>> viterbi_scores, alignments = aligner(
    ...        log_posteriors, lens, phns, phn_lens, 'viterbi'
    ... )
    >>> alignments
    [[0, 1, 2], [0, 1]]
    >>> viterbi_scores.shape
    torch.Size([2])
    """

    def __init__(
        self,
        states_per_phoneme=1,
        output_folder="",
        neg_inf=-1e5,
        batch_reduction="none",
        input_len_norm=False,
        target_len_norm=False,
        lexicon_path=None,
    ):
        super().__init__()
        self.states_per_phoneme = states_per_phoneme
        self.output_folder = output_folder
        self.neg_inf = neg_inf

        self.batch_reduction = batch_reduction
        self.input_len_norm = input_len_norm
        self.target_len_norm = target_len_norm

        self.align_dict = {}
        self.lexicon_path = lexicon_path

        if self.lexicon_path is not None:
            with open(self.lexicon_path, "r") as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                if line[0] != ";":
                    start_index = i
                    break

            lexicon = {}  # {"read": {0: "r eh d", 1: "r iy d"}}
            lexicon_phones = set()
            for i in range(start_index, len(lines)):
                line = lines[i]
                word = line.split()[0]
                phones = line.split("/")[1]

                phones = "".join([p for p in phones if not p.isdigit()])

                for p in phones.split(" "):
                    lexicon_phones.add(p)

                if "~" in word:
                    word = word.split("~")[0]
                if word in lexicon:
                    number_of_existing_pronunciations = len(lexicon[word])
                    lexicon[word][number_of_existing_pronunciations] = phones
                else:
                    lexicon[word] = {0: phones}
            self.lexicon = lexicon

            lexicon_phones = list(lexicon_phones)
            lexicon_phones.sort()

            self.lex_lab2ind = {p: i + 1 for i, p in enumerate(lexicon_phones)}
            self.lex_ind2lab = {i + 1: p for i, p in enumerate(lexicon_phones)}

            # add sil, which is not in the lexicon
            self.lex_lab2ind["sil"] = 0
            self.lex_ind2lab[0] = "sil"

    def _use_lexicon(self, words, interword_sils, sample_pron):
        """Do processing using the lexicon to return a sequence of the possible
        phonemes, the transition/pi probabilities, and the possible final states.
        Inputs correspond to a single utterance, not a whole batch.

        Arguments
        ---------
        words : list
            List of the words in the transcript.
        interword_sils : bool
            If True, optional silences will be inserted between every word.
            If False, optional silences will only be placed at the beginning
            and end of each utterance.
        sample_pron : bool
            If True, it will sample a single possible sequence of phonemes.
            If False, it will return statistics for all possible sequences of
            phonemes.

        Returns
        -------
        poss_phns : paddle.Tensor (phoneme)
            The phonemes that are thought to be in each utterance.
        log_transition_matrix : paddle.Tensor (batch, from, to)
            Tensor containing transition (log) probabilities.
        start_states : list of ints
            A list of the possible starting states in each utterance.
        final_states : list of ints
            A list of the possible final states for each utterance.
        """

        number_of_states = 0
        words_prime = (
            []
        )  # This will contain one "word" for each optional silence and pronunciation.
        # structure of each "word_prime":
        # [word index, [[state sequence 1], [state sequence 2]], <is this an optional silence?>]
        word_index = 0
        phoneme_indices = []
        for word in words:
            if word_index == 0 or interword_sils is True:
                # optional silence
                word_prime = [
                    word_index,
                    [
                        [
                            number_of_states + i
                            for i in range(self.states_per_phoneme)
                        ]
                    ],
                    True,
                ]
                words_prime.append(word_prime)
                phoneme_indices += [
                    self.silence_index * self.states_per_phoneme + i
                    for i in range(self.states_per_phoneme)
                ]
                number_of_states += self.states_per_phoneme
                word_index += 1

            # word
            word_prime = [word_index, [], False]
            if sample_pron and len(self.lexicon[word]) > 1:
                random.shuffle(self.lexicon[word])
            for pron_idx in range(len(self.lexicon[word])):
                pronunciation = self.lexicon[word][pron_idx]
                phonemes = pronunciation.split()
                word_prime[1].append([])
                for p in phonemes:
                    phoneme_indices += [
                        self.lex_lab2ind[p] * self.states_per_phoneme + i
                        for i in range(self.states_per_phoneme)
                    ]
                    word_prime[1][pron_idx] += [
                        number_of_states + i
                        for i in range(self.states_per_phoneme)
                    ]
                    number_of_states += self.states_per_phoneme
                if sample_pron:
                    break

            words_prime.append(word_prime)
            word_index += 1
        # optional final silence
        word_prime = [
            word_index,
            [[number_of_states + i for i in range(self.states_per_phoneme)]],
            True,
        ]
        words_prime.append(word_prime)
        phoneme_indices += [
            self.silence_index * self.states_per_phoneme + i
            for i in range(self.states_per_phoneme)
        ]
        number_of_states += self.states_per_phoneme
        word_index += 1

        transition_matrix = 1.0 * torch.eye(
            number_of_states
        )  # diagonal = all states have a self-loop
        final_states = []
        for word_prime in words_prime:
            word_idx = word_prime[0]
            is_optional_silence = word_prime[-1]
            next_word_exists = word_idx < len(words_prime) - 2
            this_word_last_states = [
                word_prime[1][i][-1] for i in range(len(word_prime[1]))
            ]

            # create transitions to next state from previous state within each pronunciation
            for pronunciation in word_prime[1]:
                for state_idx in range(len(pronunciation) - 1):
                    state = pronunciation[state_idx]
                    next_state = pronunciation[state_idx + 1]
                    transition_matrix[state, next_state] = 1.0

            # create transitions to next word's starting states
            if next_word_exists:
                if is_optional_silence or not interword_sils:
                    next_word_idx = word_idx + 1
                else:
                    next_word_idx = word_idx + 2
                next_word_starting_states = [
                    words_prime[next_word_idx][1][i][0]
                    for i in range(len(words_prime[next_word_idx][1]))
                ]

                for this_word_last_state in this_word_last_states:
                    for next_word_starting_state in next_word_starting_states:
                        transition_matrix[
                            this_word_last_state, next_word_starting_state
                        ] = 1.0

            else:
                final_states += this_word_last_states

            if not is_optional_silence:
                next_silence_idx = word_idx + 1
                next_silence_starting_state = words_prime[next_silence_idx][1][
                    0
                ][0]
                for this_word_last_state in this_word_last_states:
                    transition_matrix[
                        this_word_last_state, next_silence_starting_state
                    ] = 1.0

        log_transition_matrix = transition_matrix.log().log_softmax(1)

        start_states = [words_prime[0][1][0][0]]
        start_states += [
            words_prime[1][1][i][0] for i in range(len(words_prime[1][1]))
        ]

        poss_phns = torch.tensor(phoneme_indices)

        return poss_phns, log_transition_matrix, start_states, final_states

    def use_lexicon(self, words, interword_sils=True, sample_pron=False):
        """Do processing using the lexicon to return a sequence of the possible
        phonemes, the transition/pi probabilities, and the possible final
        states.
        Does processing on an utterance-by-utterance basis. Each utterance
        in the batch is processed by a helper method `_use_lexicon`.

        Arguments
        ---------
        words : list
            List of the words in the transcript
        interword_sils : bool
            If True, optional silences will be inserted between every word.
            If False, optional silences will only be placed at the beginning
            and end of each utterance.
        sample_pron: bool
            If True, it will sample a single possible sequence of phonemes.
            If False, it will return statistics for all possible sequences of
            phonemes.

        Returns
        -------
        poss_phns: paddle.Tensor (batch, phoneme in possible phn sequence)
            The phonemes that are thought to be in each utterance.
        poss_phn_lens: paddle.Tensor (batch)
            The relative length of each possible phoneme sequence in the batch.
        trans_prob: paddle.Tensor (batch, from, to)
            Tensor containing transition (log) probabilities.
        pi_prob: paddle.Tensor (batch, state)
            Tensor containing initial (log) probabilities.
        final_state: list of lists of ints
            A list of lists of possible final states for each utterance.

        Example
        -------
        >>> aligner = HMMAligner()
        >>> aligner.lexicon = {
        ...                     "a": {0: "a"},
        ...                     "b": {0: "b", 1: "c"}
        ...                   }
        >>> words = [["a", "b"]]
        >>> aligner.lex_lab2ind = {
        ...                   "sil": 0,
        ...                   "a":  1,
        ...                   "b":  2,
        ...                   "c":  3,
        ...                 }
        >>> poss_phns, poss_phn_lens, trans_prob, pi_prob, final_states = aligner.use_lexicon(
        ...     words,
        ...     interword_sils = True
        ... )
        >>> poss_phns
        tensor([[0, 1, 0, 2, 3, 0]])
        >>> poss_phn_lens
        tensor([1.])
        >>> trans_prob
        tensor([[[-6.9315e-01, -6.9315e-01, -1.0000e+05, -1.0000e+05, -1.0000e+05,
                  -1.0000e+05],
                 [-1.0000e+05, -1.3863e+00, -1.3863e+00, -1.3863e+00, -1.3863e+00,
                  -1.0000e+05],
                 [-1.0000e+05, -1.0000e+05, -1.0986e+00, -1.0986e+00, -1.0986e+00,
                  -1.0000e+05],
                 [-1.0000e+05, -1.0000e+05, -1.0000e+05, -6.9315e-01, -1.0000e+05,
                  -6.9315e-01],
                 [-1.0000e+05, -1.0000e+05, -1.0000e+05, -1.0000e+05, -6.9315e-01,
                  -6.9315e-01],
                 [-1.0000e+05, -1.0000e+05, -1.0000e+05, -1.0000e+05, -1.0000e+05,
                   0.0000e+00]]])
        >>> pi_prob
        tensor([[-6.9315e-01, -6.9315e-01, -1.0000e+05, -1.0000e+05, -1.0000e+05,
                 -1.0000e+05]])
        >>> final_states
        [[3, 4, 5]]
        >>> # With no optional silences between words
        >>> poss_phns_, _, trans_prob_, pi_prob_, final_states_ = aligner.use_lexicon(
        ...     words,
        ...     interword_sils = False
        ... )
        >>> poss_phns_
        tensor([[0, 1, 2, 3, 0]])
        >>> trans_prob_
        tensor([[[-6.9315e-01, -6.9315e-01, -1.0000e+05, -1.0000e+05, -1.0000e+05],
                 [-1.0000e+05, -1.0986e+00, -1.0986e+00, -1.0986e+00, -1.0000e+05],
                 [-1.0000e+05, -1.0000e+05, -6.9315e-01, -1.0000e+05, -6.9315e-01],
                 [-1.0000e+05, -1.0000e+05, -1.0000e+05, -6.9315e-01, -6.9315e-01],
                 [-1.0000e+05, -1.0000e+05, -1.0000e+05, -1.0000e+05,  0.0000e+00]]])
        >>> pi_prob_
        tensor([[-6.9315e-01, -6.9315e-01, -1.0000e+05, -1.0000e+05, -1.0000e+05]])
        >>> final_states_
        [[2, 3, 4]]
        >>> # With sampling of a single possible pronunciation
        >>> import random
        >>> random.seed(0)
        >>> poss_phns_, _, trans_prob_, pi_prob_, final_states_ = aligner.use_lexicon(
        ...     words,
        ...     sample_pron = True
        ... )
        >>> poss_phns_
        tensor([[0, 1, 0, 2, 0]])
        >>> trans_prob_
        tensor([[[-6.9315e-01, -6.9315e-01, -1.0000e+05, -1.0000e+05, -1.0000e+05],
                 [-1.0000e+05, -1.0986e+00, -1.0986e+00, -1.0986e+00, -1.0000e+05],
                 [-1.0000e+05, -1.0000e+05, -6.9315e-01, -6.9315e-01, -1.0000e+05],
                 [-1.0000e+05, -1.0000e+05, -1.0000e+05, -6.9315e-01, -6.9315e-01],
                 [-1.0000e+05, -1.0000e+05, -1.0000e+05, -1.0000e+05,  0.0000e+00]]])
        """
        self.silence_index = self.lex_lab2ind["sil"]

        poss_phns = []
        trans_prob = []
        start_states = []
        final_states = []

        for words_ in words:
            (
                poss_phns_,
                trans_prob_,
                start_states_,
                final_states_,
            ) = self._use_lexicon(words_, interword_sils, sample_pron)
            poss_phns.append(poss_phns_)
            trans_prob.append(trans_prob_)
            start_states.append(start_states_)
            final_states.append(final_states_)

        # pad poss_phns, trans_prob with 0 to have same length
        poss_phn_lens = [len(poss_phns_) for poss_phns_ in poss_phns]
        U_max = max(poss_phn_lens)

        batch_size = len(poss_phns)
        for index in range(batch_size):
            phn_pad_length = U_max - len(poss_phns[index])
            poss_phns[index] = torch.nn.functional.pad(
                poss_phns[index], (0, phn_pad_length), value=0
            )
            trans_prob[index] = torch.nn.functional.pad(
                trans_prob[index],
                (0, phn_pad_length, 0, phn_pad_length),
                value=self.neg_inf,
            )

        # Stack into single tensor
        poss_phns = torch.stack(poss_phns)
        trans_prob = torch.stack(trans_prob)
        trans_prob[trans_prob == -float("Inf")] = self.neg_inf

        # make pi prob
        pi_prob = self.neg_inf * torch.ones([batch_size, U_max])
        for start_state in start_states:
            pi_prob[:, start_state] = 1

        pi_prob = torch.nn.functional.log_softmax(pi_prob, dim=1)

        # Convert poss_phn_lens from absolute to relative lengths
        poss_phn_lens = torch.tensor(poss_phn_lens).float() / U_max
        return poss_phns, poss_phn_lens, trans_prob, pi_prob, final_states

    def _make_pi_prob(self, phn_lens_abs):
        """Creates tensor of initial (log) probabilities (known as 'pi').
        Assigns all probability mass to the first phoneme in the sequence.

        Arguments
        ---------
        phn_lens_abs : paddle.Tensor (batch)
            The absolute length of each phoneme sequence in the batch.

        Returns
        -------
        pi_prob : paddle.Tensor (batch, phn)
        """
        batch_size = len(phn_lens_abs)
        U_max = int(phn_lens_abs.max())

        pi_prob = self.neg_inf * torch.ones([batch_size, U_max])
        pi_prob[:, 0] = 0

        return pi_prob

    def _make_trans_prob(self, phn_lens_abs):
        """Creates tensor of transition (log) probabilities.
        Only allows transitions to the same phoneme (self-loop) or the next
        phoneme in the phn sequence

        Arguments
        ---------
        phn_lens_abs : paddle.Tensor (batch)
            The absolute length of each phoneme sequence in the batch.

        Returns
        -------
        trans_prob : paddle.Tensor (batch, from, to)
        """
        # Extract useful values for later
        batch_size = len(phn_lens_abs)
        U_max = int(phn_lens_abs.max())
        device = phn_lens_abs.device

        ## trans_prob matrix consists of 2 diagonals:
        ## (1) offset diagonal (next state) &
        ## (2) main diagonal (self-loop)
        # make offset diagonal
        trans_prob_off_diag = torch.eye(U_max - 1)
        zero_side = torch.zeros([U_max - 1, 1])
        zero_bottom = torch.zeros([1, U_max])
        trans_prob_off_diag = torch.cat((zero_side, trans_prob_off_diag), 1)
        trans_prob_off_diag = torch.cat((trans_prob_off_diag, zero_bottom), 0)

        # make main diagonal
        trans_prob_main_diag = torch.eye(U_max)

        # join the diagonals and repeat for whole batch
        trans_prob = trans_prob_off_diag + trans_prob_main_diag
        trans_prob = (
            trans_prob.reshape(1, U_max, U_max)
            .repeat(batch_size, 1, 1)
            .to(device)
        )

        # clear probabilities for too-long sequences
        mask_a = (
            torch.arange(U_max, )[None, :] < phn_lens_abs[:, None]
        )
        mask_a = mask_a.unsqueeze(2)
        mask_a = mask_a.expand(-1, -1, U_max)
        mask_b = mask_a.permute(0, 2, 1)
        trans_prob = trans_prob * (mask_a & mask_b).float()

        ## put -infs in place of zeros:
        trans_prob = torch.where(
            trans_prob == 1,
            trans_prob,
            torch.tensor(-float("Inf"), ),
        )

        ## normalize
        trans_prob = torch.nn.functional.log_softmax(trans_prob, dim=2)

        ## set nans to v neg numbers
        trans_prob[trans_prob != trans_prob] = self.neg_inf
        ## set -infs to v neg numbers
        trans_prob[trans_prob == -float("Inf")] = self.neg_inf

        return trans_prob

    def _make_emiss_pred_useful(
        self, emission_pred, lens_abs, phn_lens_abs, phns
    ):
        """Creates a 'useful' form of the posterior probabilities, rearranged
        into the order of phoneme appearance in phns.

        Arguments
        ---------
        emission_pred : paddle.Tensor (batch, time, phoneme in vocabulary)
            posterior probabilities from our acoustic model
        lens_abs : paddle.Tensor (batch)
            The absolute length of each input to the acoustic model,
            i.e., the number of frames.
        phn_lens_abs : paddle.Tensor (batch)
            The absolute length of each phoneme sequence in the batch.
        phns : paddle.Tensor (batch, phoneme in phn sequence)
            The phonemes that are known/thought to be in each utterance.

        Returns
        -------
        emiss_pred_useful : paddle.Tensor
            Tensor shape (batch, phoneme in phn sequence, time).
        """
        # Extract useful values for later
        U_max = int(phn_lens_abs.max().item())
        fb_max_length = int(lens_abs.max().item())
        device = emission_pred.device

        # apply mask based on lens_abs
        mask_lens = (
            torch.arange(fb_max_length).to(device)[None, :] < lens_abs[:, None]
        )

        emiss_pred_acc_lens = torch.where(
            mask_lens[:, :, None],
            emission_pred,
            torch.tensor([0.0], ),
        )

        # manipulate phn tensor, and then 'torch.gather'
        phns = phns.to(device)
        phns_copied = phns.unsqueeze(1).expand(-1, fb_max_length, -1)
        emiss_pred_useful = torch.gather(emiss_pred_acc_lens, 2, phns_copied)

        # apply mask based on phn_lens_abs
        mask_phn_lens = (
            torch.arange(U_max).to(device)[None, :] < phn_lens_abs[:, None]
        )
        emiss_pred_useful = torch.where(
            mask_phn_lens[:, None, :],
            emiss_pred_useful,
            torch.tensor([self.neg_inf], ),
        )

        emiss_pred_useful = emiss_pred_useful.permute(0, 2, 1)

        return emiss_pred_useful

    def _dp_forward(
        self,
        pi_prob,
        trans_prob,
        emiss_pred_useful,
        lens_abs,
        phn_lens_abs,
        phns,
    ):
        """Does forward dynamic programming algorithm.

        Arguments
        ---------
        pi_prob : paddle.Tensor (batch, phn)
            Tensor containing initial (log) probabilities.
        trans_prob : paddle.Tensor (batch, from, to)
            Tensor containing transition (log) probabilities.
        emiss_pred_useful : paddle.Tensor (batch, phoneme in phn sequence, time)
            A 'useful' form of the posterior probabilities, rearranged
            into the order of phoneme appearance in phns.
        lens_abs : paddle.Tensor (batch)
            The absolute length of each input to the acoustic model,
            i.e., the number of frames.
        phn_lens_abs : paddle.Tensor (batch)
            The absolute length of each phoneme sequence in the batch.
        phns : paddle.Tensor (batch, phoneme in phn sequence)
            The phonemes that are known/thought to be in each utterance.

        Returns
        -------
        sum_alpha_T : paddle.Tensor (batch)
            The (log) likelihood of each utterance in the batch.
        """
        # useful values
        batch_size = len(phn_lens_abs)
        U_max = phn_lens_abs.max()
        fb_max_length = lens_abs.max()
        device = emiss_pred_useful.device

        pi_prob = pi_prob.to(device)
        trans_prob = trans_prob.to(device)

        # initialise
        alpha_matrix = self.neg_inf * torch.ones(
            [batch_size, U_max, fb_max_length], 
        )
        alpha_matrix[:, :, 0] = pi_prob + emiss_pred_useful[:, :, 0]

        for t in range(1, fb_max_length):
            utt_lens_passed = lens_abs < t

            if True in utt_lens_passed:
                n_passed = utt_lens_passed.sum()
                I_tensor = self.neg_inf * torch.ones(n_passed, U_max, U_max)
                I_tensor[:, torch.arange(U_max), torch.arange(U_max)] = 0.0
                I_tensor = I_tensor.to(device)

                trans_prob[utt_lens_passed] = I_tensor

            alpha_times_trans = batch_log_matvecmul(
                trans_prob.permute(0, 2, 1), alpha_matrix[:, :, t - 1]
            )
            alpha_matrix[:, :, t] = (
                alpha_times_trans + emiss_pred_useful[:, :, t]
            )

        sum_alpha_T = torch.logsumexp(
            alpha_matrix[torch.arange(batch_size), :, -1], dim=1
        )

        return sum_alpha_T

    def _dp_viterbi(
        self,
        pi_prob,
        trans_prob,
        emiss_pred_useful,
        lens_abs,
        phn_lens_abs,
        phns,
        final_states,
    ):
        """Calculates Viterbi alignment using dynamic programming.

        Arguments
        ---------
        pi_prob : paddle.Tensor (batch, phn)
            Tensor containing initial (log) probabilities.
        trans_prob : paddle.Tensor (batch, from, to)
            Tensor containing transition (log) probabilities.
        emiss_pred_useful : paddle.Tensor (batch, phoneme in phn sequence, time)
            A 'useful' form of the posterior probabilities, rearranged
            into the order of phoneme appearance in phns.
        lens_abs : paddle.Tensor (batch)
            The absolute length of each input to the acoustic model,
            i.e., the number of frames.
        phn_lens_abs : paddle.Tensor (batch)
            The absolute length of each phoneme sequence in the batch.
        phns : paddle.Tensor (batch, phoneme in phn sequence)
            The phonemes that are known/thought to be in each utterance.

        Returns
        -------
        z_stars : list of lists of int
            Viterbi alignments for the files in the batch.
        z_stars_loc : list of lists of int
            The locations of the Viterbi alignments for the files in the batch.
            e.g., for a batch with a single utterance with 5 phonemes,
            `z_stars_loc` will look like:
            [[0, 0, 0, 1, 1, 2, 3, 3, 3, 4, 4]].
        viterbi_scores : paddle.Tensor (batch)
            The (log) likelihood of the Viterbi path for each utterance.
        """

        # useful values
        batch_size = len(phn_lens_abs)
        U_max = phn_lens_abs.max()
        fb_max_length = lens_abs.max()
        device = emiss_pred_useful.device

        pi_prob = pi_prob.to(device)
        trans_prob = trans_prob.to(device)

        v_matrix = self.neg_inf * torch.ones(
            [batch_size, U_max, fb_max_length], 
        )
        backpointers = -99 * torch.ones(
            [batch_size, U_max, fb_max_length], 
        )

        # initialise
        v_matrix[:, :, 0] = pi_prob + emiss_pred_useful[:, :, 0]

        for t in range(1, fb_max_length):
            x, argmax = batch_log_maxvecmul(
                trans_prob.permute(0, 2, 1), v_matrix[:, :, t - 1]
            )
            v_matrix[:, :, t] = x + emiss_pred_useful[:, :, t]

            backpointers[:, :, t] = argmax.type(torch.FloatTensor)

        z_stars = []
        z_stars_loc = []

        for utterance_in_batch in range(batch_size):
            len_abs = lens_abs[utterance_in_batch]

            if final_states is not None:
                final_states_utter = final_states[utterance_in_batch]
                # Pick most probable of the final states
                viterbi_finals = v_matrix[
                    utterance_in_batch, final_states_utter, len_abs - 1
                ]
                final_state_chosen = torch.argmax(viterbi_finals).item()
                U = final_states_utter[final_state_chosen]
            else:
                U = phn_lens_abs[utterance_in_batch].long().item() - 1

            z_star_i_loc = [U]
            z_star_i = [phns[utterance_in_batch, z_star_i_loc[0]].item()]
            for time_step in range(len_abs, 1, -1):
                current_best_loc = z_star_i_loc[0]

                earlier_best_loc = (
                    backpointers[
                        utterance_in_batch, current_best_loc, time_step - 1
                    ]
                    .long()
                    .item()
                )
                earlier_z_star = phns[
                    utterance_in_batch, earlier_best_loc
                ].item()

                z_star_i_loc.insert(0, earlier_best_loc)
                z_star_i.insert(0, earlier_z_star)
            z_stars.append(z_star_i)
            z_stars_loc.append(z_star_i_loc)

        # picking out viterbi_scores
        viterbi_scores = v_matrix[
            torch.arange(batch_size), phn_lens_abs - 1, lens_abs - 1
        ]

        return z_stars, z_stars_loc, viterbi_scores

    def _loss_reduction(self, loss, input_lens, target_lens):
        """Applies reduction to loss as specified during object initialization.

        Arguments
        ---------
        loss : paddle.Tensor (batch)
            The loss tensor to be reduced.
        input_lens : paddle.Tensor (batch)
            The absolute durations of the inputs.
        target_lens : paddle.Tensor (batch)
            The absolute durations of the targets.

        Returns
        -------
        loss : paddle.Tensor (batch, or scalar)
            The loss with reduction applied if it is specified.

        """
        if self.input_len_norm is True:
            loss = torch.div(loss, input_lens)

        if self.target_len_norm is True:
            loss = torch.div(loss, target_lens)

        if self.batch_reduction == "none":
            pass
        elif self.batch_reduction == "sum":
            loss = loss.sum()
        elif self.batch_reduction == "mean":
            loss = loss.mean()
        else:
            raise ValueError(
                "`batch_reduction` parameter must be one of 'none', 'sum' or 'mean'"
            )

        return loss

    def forward(
        self,
        emission_pred,
        lens,
        phns,
        phn_lens,
        dp_algorithm,
        prob_matrices=None,
    ):
        """Prepares relevant (log) probability tensors and does dynamic
        programming: either the forward or the Viterbi algorithm. Applies
        reduction as specified during object initialization.

        Arguments
        ---------
        emission_pred : paddle.Tensor (batch, time, phoneme in vocabulary)
            Posterior probabilities from our acoustic model.
        lens : paddle.Tensor (batch)
            The relative duration of each utterance sound file.
        phns : paddle.Tensor (batch, phoneme in phn sequence)
            The phonemes that are known/thought to be in each utterance
        phn_lens : paddle.Tensor (batch)
            The relative length of each phoneme sequence in the batch.
        dp_algorithm : string
            Either "forward" or "viterbi".
        prob_matrices : dict
            (Optional) Must contain keys 'trans_prob', 'pi_prob' and 'final_states'.
            Used to override the default forward and viterbi operations which
            force traversal over all of the states in the `phns` sequence.

        Returns
        -------
        tensor

            (1) if dp_algorithm == "forward".

                ``forward_scores`` : paddle.Tensor (batch, or scalar)

                The (log) likelihood of each utterance in the batch, with reduction
                applied if specified. (OR)

            (2) if dp_algorithm == "viterbi".

                ``viterbi_scores`` : paddle.Tensor (batch, or scalar)

                The (log) likelihood of the Viterbi path for each utterance, with
                reduction applied if specified.

                ``alignments`` : list of lists of int

                Viterbi alignments for the files in the batch.
        """

        lens_abs = torch.round(emission_pred.shape[1] * lens).long()
        phn_lens_abs = torch.round(phns.shape[1] * phn_lens).long()
        phns = phns.long()

        if prob_matrices is None:
            pi_prob = self._make_pi_prob(phn_lens_abs)
            trans_prob = self._make_trans_prob(phn_lens_abs)
            final_states = None
        else:
            if (
                ("pi_prob" in prob_matrices)
                and ("trans_prob" in prob_matrices)
                and ("final_states" in prob_matrices)
            ):
                pi_prob = prob_matrices["pi_prob"]
                trans_prob = prob_matrices["trans_prob"]
                final_states = prob_matrices["final_states"]
            else:
                ValueError(
                    """`prob_matrices` must contain the keys
                `pi_prob`, `trans_prob` and `final_states`"""
                )

        emiss_pred_useful = self._make_emiss_pred_useful(
            emission_pred, lens_abs, phn_lens_abs, phns
        )

        if dp_algorithm == "forward":
            # do forward training
            forward_scores = self._dp_forward(
                pi_prob,
                trans_prob,
                emiss_pred_useful,
                lens_abs,
                phn_lens_abs,
                phns,
            )

            forward_scores = self._loss_reduction(
                forward_scores, lens_abs, phn_lens_abs
            )

            return forward_scores

        elif dp_algorithm == "viterbi":
            alignments, _, viterbi_scores = self._dp_viterbi(
                pi_prob,
                trans_prob,
                emiss_pred_useful,
                lens_abs,
                phn_lens_abs,
                phns,
                final_states,
            )

            viterbi_scores = self._loss_reduction(
                viterbi_scores, lens_abs, phn_lens_abs
            )

            return viterbi_scores, alignments

        else:
            raise ValueError(
                "dp_algorithm input must be either 'forward' or 'viterbi'"
            )

    def expand_phns_by_states_per_phoneme(self, phns, phn_lens):
        """Expands each phoneme in the phn sequence by the number of hidden
        states per phoneme defined in the HMM.

        Arguments
        ---------
        phns : paddle.Tensor (batch, phoneme in phn sequence)
            The phonemes that are known/thought to be in each utterance.
        phn_lens : paddle.Tensor (batch)
            The relative length of each phoneme sequence in the batch.

        Returns
        -------
        expanded_phns : paddle.Tensor (batch, phoneme in expanded phn sequence)

        Example
        -------
        >>> phns = torch.tensor([[0., 3., 5., 0.],
        ...                      [0., 2., 0., 0.]])
        >>> phn_lens = torch.tensor([1., 0.75])
        >>> aligner = HMMAligner(states_per_phoneme = 3)
        >>> expanded_phns = aligner.expand_phns_by_states_per_phoneme(
        ...         phns, phn_lens
        ... )
        >>> expanded_phns
        tensor([[ 0.,  1.,  2.,  9., 10., 11., 15., 16., 17.,  0.,  1.,  2.],
                [ 0.,  1.,  2.,  6.,  7.,  8.,  0.,  1.,  2.,  0.,  0.,  0.]])
        """
        # Initialise expanded_phns
        expanded_phns = torch.zeros(
            phns.shape[0], phns.shape[1] * self.states_per_phoneme
        )
        expanded_phns = expanded_phns.to(phns.device)

        phns = undo_padding(phns, phn_lens)
        for i, phns_utt in enumerate(phns):
            expanded_phns_utt = []
            for phoneme_index in phns_utt:
                expanded_phns_utt += [
                    self.states_per_phoneme * phoneme_index + i_
                    for i_ in range(self.states_per_phoneme)
                ]

            expanded_phns[i, : len(expanded_phns_utt)] = torch.tensor(
                expanded_phns_utt
            )
        return expanded_phns

    def store_alignments(self, ids, alignments):
        """Records Viterbi alignments in `self.align_dict`.

        Arguments
        ---------
        ids : list of str
            IDs of the files in the batch.
        alignments : list of lists of int
            Viterbi alignments for the files in the batch.
            Without padding.

        Example
        -------
        >>> aligner = HMMAligner()
        >>> ids = ['id1', 'id2']
        >>> alignments = [[0, 2, 4], [1, 2, 3, 4]]
        >>> aligner.store_alignments(ids, alignments)
        >>> aligner.align_dict.keys()
        dict_keys(['id1', 'id2'])
        >>> aligner.align_dict['id1']
        tensor([0, 2, 4], dtype=torch.int16)
        """

        for i, id in enumerate(ids):
            alignment_i = alignments[i]
            alignment_i = torch.tensor(alignment_i, dtype=torch.int16).cpu()
            self.align_dict[id] = alignment_i

    def _get_flat_start_batch(self, lens_abs, phn_lens_abs, phns):
        """Prepares flat start alignments (with zero padding) for every utterance
        in the batch.
        Every phoneme will have an equal duration, except for the final phoneme
        potentially. E.g. if 104 frames and 10 phonemes, 9 phonemes will have
        duration of 10 frames, and one phoneme will have a duration of 14 frames.

        Arguments
        ---------
        lens_abs : paddle.Tensor (batch)
            The absolute length of each input to the acoustic model,
            i.e., the number of frames.

        phn_lens_abs : paddle.Tensor (batch)
            The absolute length of each phoneme sequence in the batch.

        phns : paddle.Tensor (batch, phoneme in phn sequence)
            The phonemes that are known/thought to be in each utterance.

        Returns
        -------
        flat_start_batch : paddle.Tensor (batch, time)
            Flat start alignments for utterances in the batch, with zero padding.
        """
        phns = phns.long()

        batch_size = len(lens_abs)
        fb_max_length = torch.max(lens_abs)

        flat_start_batch = torch.zeros(
            batch_size, fb_max_length, device=phns.device
        ).long()
        for i in range(batch_size):
            utter_phns = phns[i]
            utter_phns = utter_phns[: phn_lens_abs[i]]  # crop out zero padding
            repeat_amt = int(lens_abs[i].item() / len(utter_phns))

            # make sure repeat_amt is at least 1. (the code above
            # may make repeat_amt==0 if self.states_per_phoneme is too large).
            if repeat_amt == 0:
                repeat_amt = 1

            # repeat each phoneme in utter_phns by repeat_amt
            utter_phns = utter_phns.repeat_interleave(repeat_amt)

            # len(utter_phns) may be <, == or > lens_abs[i], so
            # make sure len(utter_phns) == lens_abs[i]
            utter_phns = utter_phns[: lens_abs[i]]
            utter_phns = torch.nn.functional.pad(
                utter_phns,
                (0, int(lens_abs[i]) - len(utter_phns)),
                value=utter_phns[-1],  # pad out with final phoneme
            )

            flat_start_batch[i, : len(utter_phns)] = utter_phns

        return flat_start_batch

    def _get_viterbi_batch(self, ids, lens_abs):
        """Retrieves Viterbi alignments stored in `self.align_dict` and
        creates a batch of them, with zero padding.

        Arguments
        ---------
        ids : list of str
            IDs of the files in the batch.
        lens_abs : paddle.Tensor (batch)
            The absolute length of each input to the acoustic model,
            i.e., the number of frames.

        Returns
        -------
        viterbi_batch : paddle.Tensor (batch, time)
            The previously-recorded Viterbi alignments for the utterances
            in the batch.

        """
        batch_size = len(lens_abs)
        fb_max_length = torch.max(lens_abs)

        viterbi_batch = torch.zeros(
            batch_size, fb_max_length, device=lens_abs.device
        ).long()
        for i in range(batch_size):
            viterbi_preds = self.align_dict[ids[i]]
            viterbi_preds = torch.nn.functional.pad(
                viterbi_preds, (0, fb_max_length - len(viterbi_preds))
            )

            viterbi_batch[i] = viterbi_preds.long()

        return viterbi_batch

    def get_prev_alignments(self, ids, emission_pred, lens, phns, phn_lens):
        """Fetches previously recorded Viterbi alignments if they are available.
        If not, fetches flat start alignments.
        Currently, assumes that if a Viterbi alignment is not available for the
        first utterance in the batch, it will not be available for the rest of
        the utterances.

        Arguments
        ---------
        ids : list of str
            IDs of the files in the batch.
        emission_pred : paddle.Tensor (batch, time, phoneme in vocabulary)
            Posterior probabilities from our acoustic model. Used to infer the
            duration of the longest utterance in the batch.
        lens : paddle.Tensor (batch)
            The relative duration of each utterance sound file.
        phns : paddle.Tensor (batch, phoneme in phn sequence)
            The phonemes that are known/thought to be in each utterance.
        phn_lens : paddle.Tensor (batch)
            The relative length of each phoneme sequence in the batch.

        Returns
        -------
        paddle.Tensor (batch, time)
            Zero-padded alignments.

        Example
        -------
        >>> ids = ['id1', 'id2']
        >>> emission_pred = torch.tensor([[[ -1., -10., -10.],
        ...                                [-10.,  -1., -10.],
        ...                                [-10., -10.,  -1.]],
        ...
        ...                               [[ -1., -10., -10.],
        ...                                [-10.,  -1., -10.],
        ...                                [-10., -10., -10.]]])
        >>> lens = torch.tensor([1., 0.66])
        >>> phns = torch.tensor([[0, 1, 2],
        ...                      [0, 1, 0]])
        >>> phn_lens = torch.tensor([1., 0.66])
        >>> aligner = HMMAligner()
        >>> alignment_batch = aligner.get_prev_alignments(
        ...        ids, emission_pred, lens, phns, phn_lens
        ... )
        >>> alignment_batch
        tensor([[0, 1, 2],
                [0, 1, 0]])
        """

        lens_abs = torch.round(emission_pred.shape[1] * lens).long()
        phn_lens_abs = torch.round(phns.shape[1] * phn_lens).long()

        if ids[0] in self.align_dict:
            return self._get_viterbi_batch(ids, lens_abs)
        else:
            return self._get_flat_start_batch(lens_abs, phn_lens_abs, phns)

    def _calc_accuracy_sent(self, alignments_, ends_, phns_):
        """Calculates the accuracy between predicted alignments and ground truth
        alignments for a single sentence/utterance.

        Arguments
        ---------
        alignments_ : list of ints
            The predicted alignments for the utterance.
        ends_ : list of ints
            A list of the sample indices where each ground truth phoneme
            ends, according to the transcription.
        phns_ : list of ints
            The unpadded list of ground truth phonemes in the utterance.

        Returns
        -------
        mean_acc : float
            The mean percentage of times that the upsampled predicted alignment
            matches the ground truth alignment.
        """
        # Create array containing the true alignment at each sample
        ends_ = [0] + [int(end) for end in ends_]
        true_durations = [ends_[i] - ends_[i - 1] for i in range(1, len(ends_))]
        true_alignments = []

        for i in range(len(phns_)):
            true_alignments += [phns_[i]] * (true_durations[i])
        true_alignments = torch.tensor(true_alignments)

        # Upsample the predicted alignment array
        # and make sure length matches that of `true_alignment`
        upsample_factor = int(
            torch.round(torch.tensor(len(true_alignments) / len(alignments_)))
        )

        alignments_ = torch.tensor(alignments_)
        alignments_upsampled = alignments_.repeat_interleave(upsample_factor)
        alignments_upsampled = alignments_upsampled[: len(true_alignments)]

        if len(true_alignments) > len(alignments_upsampled):
            alignments_upsampled = torch.nn.functional.pad(
                alignments_upsampled,
                (0, len(true_alignments) - len(alignments_upsampled)),
            )

        # Measure sample-wise accuracy
        accuracy = (
            alignments_upsampled == true_alignments
        ).float().mean().item() * 100

        return accuracy

    def calc_accuracy(self, alignments, ends, phns, ind2labs=None):
        """Calculates mean accuracy between predicted alignments and ground truth
        alignments. Ground truth alignments are derived from ground truth phns
        and their ends in the audio sample.

        Arguments
        ---------
        alignments : list of lists of ints/floats
            The predicted alignments for each utterance in the batch.
        ends : list of lists of ints
            A list of lists of sample indices where each ground truth phoneme
            ends, according to the transcription.
            Note: current implementation assumes that 'ends' mark the index
            where the next phoneme begins.
        phns : list of lists of ints/floats
            The unpadded list of lists of ground truth phonemes in the batch.
        ind2labs : tuple
            (Optional)
            Contains the original index-to-label dicts for the first and second
            sequence of phonemes.

        Returns
        -------
        mean_acc : float
            The mean percentage of times that the upsampled predicted alignment
            matches the ground truth alignment.

        Example
        -------
        >>> aligner = HMMAligner()
        >>> alignments = [[0., 0., 0., 1.]]
        >>> phns = [[0., 1.]]
        >>> ends = [[2, 4]]
        >>> mean_acc = aligner.calc_accuracy(alignments, ends, phns)
        >>> mean_acc.item()
        75.0
        """
        acc_hist = []

        # Do conversion if states_per_phoneme > 1
        if self.states_per_phoneme > 1:
            alignments = [
                [i // self.states_per_phoneme for i in utt]
                for utt in alignments
            ]

        # convert to common alphabet if need be
        if ind2labs is not None:
            alignments, phns = map_inds_to_intersect(alignments, phns, ind2labs)

        for alignments_, ends_, phns_ in zip(alignments, ends, phns):
            acc = self._calc_accuracy_sent(alignments_, ends_, phns_)
            acc_hist.append(acc)

        acc_hist = torch.tensor(acc_hist)
        mean_acc = acc_hist.mean()

        return mean_acc.unsqueeze(0)

    def collapse_alignments(self, alignments):
        """
        Converts alignments to 1 state per phoneme style.

        Arguments
        ---------
        alignments : list of ints
            Predicted alignments for a single utterance.

        Returns
        -------
        sequence : list of ints
            The predicted alignments converted to a 1 state per phoneme style.

        Example
        -------
        >>> aligner = HMMAligner(states_per_phoneme = 3)
        >>> alignments = [0, 1, 2, 3, 4, 5, 3, 4, 5, 0, 1, 2]
        >>> sequence = aligner.collapse_alignments(alignments)
        >>> sequence
        [0, 1, 1, 0]
        """

        # Filter the repetitions
        sequence = [
            v
            for i, v in enumerate(alignments)
            if i == 0 or v != alignments[i - 1]
        ]

        # Pick out only multiples of self.states_per_phoneme
        sequence = [v for v in sequence if v % self.states_per_phoneme == 0]

        # Divide by self.states_per_phoneme
        sequence = [v // self.states_per_phoneme for v in sequence]

        return sequence

    @mark_as_saver
    def _save(self, path):
        torch.save(self.align_dict, path)

    @mark_as_loader
    def _load(self, path, end_of_epoch=False, device=None):
        del end_of_epoch  # Not used here.
        del device
        self.align_dict = torch.load(path)


def map_inds_to_intersect(lists1, lists2, ind2labs):
    """Converts 2 lists containing indices for phonemes from different
    phoneme sets to a single phoneme so that comparing the equality
    of the indices of the resulting lists will yield the correct
    accuracy.

    Arguments
    ---------
    lists1 : list of lists of ints
        Contains the indices of the first sequence of phonemes.
    lists2 : list of lists of ints
        Contains the indices of the second sequence of phonemes.
    ind2labs : tuple (dict, dict)
        Contains the original index-to-label dicts for the first and second
        sequence of phonemes.

    Returns
    -------
    lists1_new : list of lists of ints
        Contains the indices of the first sequence of phonemes, mapped
        to the new phoneme set.
    lists2_new : list of lists of ints
        Contains the indices of the second sequence of phonemes, mapped
        to the new phoneme set.

    Example
    -------
    >>> lists1 = [[0, 1]]
    >>> lists2 = [[0, 1]]
    >>> ind2lab1 = {
    ...        0: "a",
    ...        1: "b",
    ...        }
    >>> ind2lab2 = {
    ...        0: "a",
    ...        1: "c",
    ...        }
    >>> ind2labs = (ind2lab1, ind2lab2)
    >>> out1, out2 = map_inds_to_intersect(lists1, lists2, ind2labs)
    >>> out1
    [[0, 1]]
    >>> out2
    [[0, 2]]
    """
    ind2lab1, ind2lab2 = ind2labs

    # Form 3 sets:
    # (1) labs in both mappings
    # (2) labs in only 1st mapping
    # (3) labs in only 2nd mapping
    set1, set2 = set(ind2lab1.values()), set(ind2lab2.values())

    intersect = set1.intersection(set2)
    set1_only = set1.difference(set2)
    set2_only = set2.difference(set1)

    new_lab2ind = {lab: i for i, lab in enumerate(intersect)}
    new_lab2ind.update(
        {lab: len(new_lab2ind) + i for i, lab in enumerate(set1_only)}
    )
    new_lab2ind.update(
        {lab: len(new_lab2ind) + i for i, lab in enumerate(set2_only)}
    )

    # Map lists to labels and apply new_lab2ind
    lists1_lab = [[ind2lab1[ind] for ind in utt] for utt in lists1]
    lists2_lab = [[ind2lab2[ind] for ind in utt] for utt in lists2]

    lists1_new = [[new_lab2ind[lab] for lab in utt] for utt in lists1_lab]
    lists2_new = [[new_lab2ind[lab] for lab in utt] for utt in lists2_lab]

    return lists1_new, lists2_new


def batch_log_matvecmul(A, b):
    """For each 'matrix' and 'vector' pair in the batch, do matrix-vector
    multiplication in the log domain, i.e., logsumexp instead of add,
    add instead of multiply.

    Arguments
    ---------
    A : paddle.Tensor (batch, dim1, dim2)
        Tensor
    b : paddle.Tensor (batch, dim1)
        Tensor.

    Outputs
    -------
    x : paddle.Tensor (batch, dim1)

    Example
    -------
    >>> A = torch.tensor([[[   0., 0.],
    ...                    [ -1e5, 0.]]])
    >>> b = torch.tensor([[0., 0.,]])
    >>> x = batch_log_matvecmul(A, b)
    >>> x
    tensor([[0.6931, 0.0000]])
    >>>
    >>> # non-log domain equivalent without batching functionality
    >>> A_ = torch.tensor([[1., 1.],
    ...                    [0., 1.]])
    >>> b_ = torch.tensor([1., 1.,])
    >>> x_ = torch.matmul(A_, b_)
    >>> x_
    tensor([2., 1.])
    """
    b = b.unsqueeze(1)
    x = torch.logsumexp(A + b, dim=2)

    return x


def batch_log_maxvecmul(A, b):
    """Similar to batch_log_matvecmul, but takes a maximum instead of
    logsumexp. Returns both the max and the argmax.

    Arguments
    ---------
    A : paddle.Tensor (batch, dim1, dim2)
        Tensor.
    b : paddle.Tensor (batch, dim1)
        Tensor

    Outputs
    -------
    x : paddle.Tensor (batch, dim1)
        Tensor.
    argmax : paddle.Tensor (batch, dim1)
        Tensor.

    Example
    -------
    >>> A = torch.tensor([[[   0., -1.],
    ...                    [ -1e5,  0.]]])
    >>> b = torch.tensor([[0., 0.,]])
    >>> x, argmax = batch_log_maxvecmul(A, b)
    >>> x
    tensor([[0., 0.]])
    >>> argmax
    tensor([[0, 1]])
    """
    b = b.unsqueeze(1)
    x, argmax = torch.max(A + b, dim=2)

    return x, argmax
