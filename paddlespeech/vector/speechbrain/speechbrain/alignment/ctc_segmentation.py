#!/usr/bin/env python3
# 2021, Technische Universität München, Ludwig Kürzinger
"""Perform CTC segmentation to align utterances within audio files.

This uses the ctc-segmentation Python package.
Install it with pip or see the installing instructions in
https://github.com/lumaku/ctc-segmentation
"""

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from typing import Union

import numpy as np
import paddle
from typing import List

# speechbrain interface
from speechbrain.pretrained.interfaces import EncoderASR, EncoderDecoderASR

# imports for CTC segmentation
try:
    from ctc_segmentation import ctc_segmentation
    from ctc_segmentation import CtcSegmentationParameters
    from ctc_segmentation import determine_utterance_segments
    from ctc_segmentation import prepare_text
    from ctc_segmentation import prepare_token_list
except ImportError:
    print(
        "ImportError: "
        "Is the ctc_segmentation module installed "
        "and in your PYTHONPATH?"
    )
    raise ImportError("The ctc_segmentation module is missing.")

logger = logging.getLogger(__name__)


class CTCSegmentationTask(SimpleNamespace):
    """Task object for CTC segmentation.

    This object is automatically generated and acts as
    a container for results of a CTCSegmentation object.

    When formatted with str(·), this object returns
    results in a kaldi-style segments file formatting.
    The human-readable output can be configured with
    the printing options.

    Properties
    ---------
    text : list
        Utterance texts, separated by line. But without the utterance
            name at the beginning of the line (as in kaldi-style text).
    ground_truth_mat : array
        Ground truth matrix (CTC segmentation).
    utt_begin_indices : np.ndarray
        Utterance separator for the Ground truth matrix.
    timings : np.ndarray
        Time marks of the corresponding chars.
    state_list : list
        Estimated alignment of chars/tokens.
    segments : list
        Calculated segments as: (start, end, confidence score).
    config : CtcSegmentationParameters
        CTC Segmentation configuration object.
    name : str
        Name of aligned audio file (Optional). If given, name is
        considered when generating the text.
        Default: "utt".
    utt_ids : list
        The list of utterance names (Optional). This list should
        have the same length as the number of utterances.
    lpz : np.ndarray
        CTC posterior log probabilities (Optional).

    Properties for printing
    ----------------------
    print_confidence_score : bool
        Include the confidence score.
        Default: True.
    print_utterance_text : bool
        Include utterance text.
        Default: True.

    """

    text = None
    ground_truth_mat = None
    utt_begin_indices = None
    timings = None
    char_probs = None
    state_list = None
    segments = None
    config = None
    done = False
    # Optional
    name = "utt"
    utt_ids = None
    lpz = None
    # Printing
    print_confidence_score = True
    print_utterance_text = True

    def set(self, **kwargs):
        """Update object attributes."""
        self.__dict__.update(kwargs)

    def __str__(self):
        """Return a kaldi-style ``segments`` file (string)."""
        output = ""
        num_utts = len(self.segments)
        if self.utt_ids is None:
            utt_names = [f"{self.name}_{i:04}" for i in range(num_utts)]
        else:
            # ensure correct mapping of segments to utterance ids
            assert num_utts == len(self.utt_ids)
            utt_names = self.utt_ids
        for i, boundary in enumerate(self.segments):
            # utterance name and file name
            utt_entry = f"{utt_names[i]} {self.name} "
            # segment start and end
            utt_entry += f"{boundary[0]:.2f} {boundary[1]:.2f}"
            # confidence score
            if self.print_confidence_score:
                utt_entry += f" {boundary[2]:3.4f}"
            # utterance ground truth
            if self.print_utterance_text:
                utt_entry += f" {self.text[i]}"
            output += utt_entry + "\n"
        return output


class CTCSegmentation:
    """Align text to audio using CTC segmentation.

    Usage
    -----
    Initialize with given ASR model and parameters.
    If needed, parameters for CTC segmentation can be set with ``set_config(·)``.
    Then call the instance as function to align text within an audio file.

    Arguments
    ---------
    asr_model : EncoderDecoderASR
        Speechbrain ASR interface. This requires a model that has a
        trained CTC layer for inference. It is better to use a model with
        single-character tokens to get a better time resolution.
        Please note that the inference complexity with Transformer models
        usually increases quadratically with audio length.
        It is therefore recommended to use RNN-based models, if available.
    kaldi_style_text : bool
        A kaldi-style text file includes the name of the
        utterance at the start of the line. If True, the utterance name
        is expected as first word at each line. If False, utterance
        names are automatically generated. Set this option according to
        your input data. Default: True.
    text_converter : str
        How CTC segmentation handles text.
        "tokenize": Use the ASR model tokenizer to tokenize the text.
        "classic": The text is preprocessed as text pieces which takes
        token length into account. If the ASR model has longer tokens,
        this option may yield better results. Default: "tokenize".
    time_stamps : str
        Choose the method how the time stamps are
        calculated. While "fixed" and "auto" use both the sample rate,
        the ratio of samples to one frame is either automatically
        determined for each inference or fixed at a certain ratio that
        is initially determined by the module, but can be changed via
        the parameter ``samples_to_frames_ratio``. Recommended for
        longer audio files: "auto".
    **ctc_segmentation_args
        Parameters for CTC segmentation.
        The full list of parameters is found in ``set_config``.

    Example
    -------
        >>> # using example file included in the SpeechBrain repository
        >>> from speechbrain.pretrained import EncoderDecoderASR
        >>> from speechbrain.alignment.ctc_segmentation import CTCSegmentation
        >>> # load an ASR model
        >>> pre_trained = "speechbrain/asr-transformer-transformerlm-librispeech"
        >>> asr_model = EncoderDecoderASR.from_hparams(source=pre_trained)
        >>> aligner = CTCSegmentation(asr_model, kaldi_style_text=False)
        >>> # load data
        >>> audio_path = "./samples/audio_samples/example1.wav"
        >>> text = ["THE BIRCH CANOE", "SLID ON THE", "SMOOTH PLANKS"]
        >>> segments = aligner(audio_path, text, name="example1")

    On multiprocessing
    ------------------
    To parallelize the computation with multiprocessing, these three steps
    can be separated:
    (1) ``get_lpz``: obtain the lpz,
    (2) ``prepare_segmentation_task``: prepare the task, and
    (3) ``get_segments``: perform CTC segmentation.
    Note that the function `get_segments` is a staticmethod and therefore
    independent of an already initialized CTCSegmentation obj́ect.

    References
    ----------
    CTC-Segmentation of Large Corpora for German End-to-end Speech Recognition
    2020, Kürzinger, Winkelbauer, Li, Watzel, Rigoll
    https://arxiv.org/abs/2007.09127

    More parameters are described in https://github.com/lumaku/ctc-segmentation

    """

    fs = 16000
    kaldi_style_text = True
    samples_to_frames_ratio = None
    time_stamps = "auto"
    choices_time_stamps = ["auto", "fixed"]
    text_converter = "tokenize"
    choices_text_converter = ["tokenize", "classic"]
    warned_about_misconfiguration = False
    config = CtcSegmentationParameters()

    def __init__(
        self,
        asr_model: Union[EncoderASR, EncoderDecoderASR],
        kaldi_style_text: bool = True,
        text_converter: str = "tokenize",
        time_stamps: str = "auto",
        **ctc_segmentation_args,
    ):
        """Initialize the CTCSegmentation module."""
        # Prepare ASR model
        if (
            isinstance(asr_model, EncoderDecoderASR)
            and not (
                hasattr(asr_model, "mods")
                and hasattr(asr_model.mods, "decoder")
                and hasattr(asr_model.mods.decoder, "ctc_weight")
            )
        ) or (
            isinstance(asr_model, EncoderASR)
            and not (
                hasattr(asr_model, "mods")
                and hasattr(asr_model.mods, "encoder")
                and hasattr(asr_model.mods.encoder, "ctc_lin")
            )
        ):
            raise AttributeError("The given asr_model has no CTC module!")
        if not hasattr(asr_model, "tokenizer"):
            raise AttributeError(
                "The given asr_model has no tokenizer in asr_model.tokenizer!"
            )
        self.asr_model = asr_model
        self._encode = self.asr_model.encode_batch
        if isinstance(asr_model, EncoderDecoderASR):
            # Assumption: log-softmax is already included in ctc_forward_step
            self._ctc = self.asr_model.mods.decoder.ctc_forward_step
        else:
            # Apply log-softmax to encoder output
            self._ctc = self.asr_model.hparams.log_softmax
        self._tokenizer = self.asr_model.tokenizer

        # Apply configuration
        self.set_config(
            fs=self.asr_model.hparams.sample_rate,
            time_stamps=time_stamps,
            kaldi_style_text=kaldi_style_text,
            text_converter=text_converter,
            **ctc_segmentation_args,
        )

        # determine token or character list
        char_list = [
            asr_model.tokenizer.id_to_piece(i)
            for i in range(asr_model.tokenizer.vocab_size())
        ]
        self.config.char_list = char_list

        # Warn about possible misconfigurations
        max_char_len = max([len(c) for c in char_list])
        if len(char_list) > 500 and max_char_len >= 8:
            logger.warning(
                f"The dictionary has {len(char_list)} tokens with "
                f"a max length of {max_char_len}. This may lead "
                f"to low alignment performance and low accuracy."
            )

    def set_config(
        self,
        time_stamps: Optional[str] = None,
        fs: Optional[int] = None,
        samples_to_frames_ratio: Optional[float] = None,
        set_blank: Optional[int] = None,
        replace_spaces_with_blanks: Optional[bool] = None,
        kaldi_style_text: Optional[bool] = None,
        text_converter: Optional[str] = None,
        gratis_blank: Optional[bool] = None,
        min_window_size: Optional[int] = None,
        max_window_size: Optional[int] = None,
        scoring_length: Optional[int] = None,
    ):
        """Set CTC segmentation parameters.

        Parameters for timing
        ---------------------
        time_stamps : str
            Select method how CTC index duration is estimated, and
            thus how the time stamps are calculated.
        fs : int
            Sample rate. Usually derived from ASR model; use this parameter
            to overwrite the setting.
        samples_to_frames_ratio : float
            If you want to directly determine the
            ratio of samples to CTC frames, set this parameter, and
            set ``time_stamps`` to "fixed".
            Note: If you want to calculate the time stamps from a model
            with fixed subsampling, set this parameter to:
            ``subsampling_factor * frame_duration / 1000``.

        Parameters for text preparation
        -------------------------------
        set_blank : int
            Index of blank in token list. Default: 0.
        replace_spaces_with_blanks : bool
            Inserts blanks between words, which is
            useful for handling long pauses between words. Only used in
            ``text_converter="classic"`` preprocessing mode. Default: False.
        kaldi_style_text : bool
            Determines whether the utterance name is expected
            as fist word of the utterance. Set at module initialization.
        text_converter : str
            How CTC segmentation handles text.
            Set at module initialization.

        Parameters for alignment
        ------------------------
        min_window_size : int
            Minimum number of frames considered for a single
            utterance. The current default value of 8000 corresponds to
            roughly 4 minutes (depending on ASR model) and should be OK in
            most cases. If your utterances are further apart, increase
            this value, or decrease it for smaller audio files.
        max_window_size : int
            Maximum window size. It should not be necessary
            to change this value.
        gratis_blank : bool
            If True, the transition cost of blank is set to zero.
            Useful for long preambles or if there are large unrelated segments
            between utterances. Default: False.

        Parameters for calculation of confidence score
        ----------------------------------------------
        scoring_length : int
            Block length to calculate confidence score. The
            default value of 30 should be OK in most cases.
            30 corresponds to roughly 1-2s of audio.
        """
        # Parameters for timing
        if time_stamps is not None:
            if time_stamps not in self.choices_time_stamps:
                raise NotImplementedError(
                    f"Parameter ´time_stamps´ has to be one of "
                    f"{list(self.choices_time_stamps)}",
                )
            self.time_stamps = time_stamps
        if fs is not None:
            self.fs = float(fs)
        if samples_to_frames_ratio is not None:
            self.samples_to_frames_ratio = float(samples_to_frames_ratio)
        # Parameters for text preparation
        if set_blank is not None:
            self.config.blank = int(set_blank)
        if replace_spaces_with_blanks is not None:
            self.config.replace_spaces_with_blanks = bool(
                replace_spaces_with_blanks
            )
        if kaldi_style_text is not None:
            self.kaldi_style_text = bool(kaldi_style_text)
        if text_converter is not None:
            if text_converter not in self.choices_text_converter:
                raise NotImplementedError(
                    f"Parameter ´text_converter´ has to be one of "
                    f"{list(self.choices_text_converter)}",
                )
            self.text_converter = text_converter
        # Parameters for alignment
        if min_window_size is not None:
            self.config.min_window_size = int(min_window_size)
        if max_window_size is not None:
            self.config.max_window_size = int(max_window_size)
        if gratis_blank is not None:
            self.config.blank_transition_cost_zero = bool(gratis_blank)
        if (
            self.config.blank_transition_cost_zero
            and self.config.replace_spaces_with_blanks
            and not self.warned_about_misconfiguration
        ):
            logger.error(
                "Blanks are inserted between words, and also the transition cost of"
                " blank is zero. This configuration may lead to misalignments!"
            )
            self.warned_about_misconfiguration = True
        # Parameter for calculation of confidence score
        if scoring_length is not None:
            self.config.score_min_mean_over_L = int(scoring_length)

    def get_timing_config(self, speech_len=None, lpz_len=None):
        """Obtain parameters to determine time stamps."""
        timing_cfg = {
            "index_duration": self.config.index_duration,
        }
        # As the parameter ctc_index_duration vetoes the other
        if self.time_stamps == "fixed":
            # Initialize the value, if not yet available
            if self.samples_to_frames_ratio is None:
                ratio = self.estimate_samples_to_frames_ratio()
                self.samples_to_frames_ratio = ratio
            index_duration = self.samples_to_frames_ratio / self.fs
        else:
            assert self.time_stamps == "auto"
            samples_to_frames_ratio = speech_len / lpz_len
            index_duration = samples_to_frames_ratio / self.fs
        timing_cfg["index_duration"] = index_duration
        return timing_cfg

    def estimate_samples_to_frames_ratio(self, speech_len=215040):
        """Determine the ratio of encoded frames to sample points.

        This method helps to determine the time a single encoded frame occupies.
        As the sample rate already gave the number of samples, only the ratio
        of samples per encoded CTC frame are needed. This function estimates them by
        doing one inference, which is only needed once.

        Args
        ----
        speech_len : int
            Length of randomly generated speech vector for single
            inference. Default: 215040.

        Returns
        -------
        int
            Estimated ratio.
        """
        random_input = torch.rand(speech_len)
        lpz = self.get_lpz(random_input)
        lpz_len = lpz.shape[0]
        # CAVEAT assumption: Frontend does not discard trailing data!
        samples_to_frames_ratio = speech_len / lpz_len
        return samples_to_frames_ratio

    @torch.no_grad()
    def get_lpz(self, speech: Union[paddle.Tensor, np.ndarray]):
        """Obtain CTC posterior log probabilities for given speech data.

        Args
        ----
        speech : Union[paddle.Tensor, np.ndarray]
            Speech audio input.

        Returns
        -------
        np.ndarray
            Numpy vector with CTC log posterior probabilities.
        """
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)
        # Batch data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(self.asr_model.device)
        wav_lens = torch.tensor([1.0]).to(self.asr_model.device)
        enc = self._encode(speech, wav_lens)
        # Apply ctc layer to obtain log character probabilities
        lpz = self._ctc(enc).detach()
        #  Shape should be ( <time steps>, <classes> )
        lpz = lpz.squeeze(0).cpu().numpy()
        return lpz

    def _split_text(self, text):
        """Convert text to list and extract utterance IDs."""
        utt_ids = None
        # Handle multiline strings
        if isinstance(text, str):
            text = text.splitlines()
        # Remove empty lines
        text = list(filter(len, text))
        # Handle kaldi-style text format
        if self.kaldi_style_text:
            utt_ids_and_text = [utt.split(" ", 1) for utt in text]
            # remove utterances with empty text
            utt_ids_and_text = filter(lambda ui: len(ui) == 2, utt_ids_and_text)
            utt_ids_and_text = list(utt_ids_and_text)
            utt_ids = [utt[0] for utt in utt_ids_and_text]
            text = [utt[1] for utt in utt_ids_and_text]
        return utt_ids, text

    def prepare_segmentation_task(self, text, lpz, name=None, speech_len=None):
        """Preprocess text, and gather text and lpz into a task object.

        Text is pre-processed and tokenized depending on configuration.
        If ``speech_len`` is given, the timing configuration is updated.
        Text, lpz, and configuration is collected in a CTCSegmentationTask
        object. The resulting object can be serialized and passed in a
        multiprocessing computation.

        It is recommended that you normalize the text beforehand, e.g.,
        change numbers into their spoken equivalent word, remove special
        characters, and convert UTF-8 characters to chars corresponding to
        your ASR model dictionary.

        The text is tokenized based on the ``text_converter`` setting:

        The "tokenize" method is more efficient and the easiest for models
        based on latin or cyrillic script that only contain the main chars,
        ["a", "b", ...] or for Japanese or Chinese ASR models with ~3000
        short Kanji / Hanzi tokens.

        The "classic" method improves the the accuracy of the alignments
        for models that contain longer tokens, but with a greater complexity
        for computation. The function scans for partial tokens which may
        improve time resolution.
        For example, the word "▁really" will be broken down into
        ``['▁', '▁r', '▁re', '▁real', '▁really']``. The alignment will be
        based on the most probable activation sequence given by the network.

        Args
        ----
        text : list
            List or multiline-string with utterance ground truths.
        lpz : np.ndarray
            Log CTC posterior probabilities obtained from the CTC-network;
            numpy array shaped as ( <time steps>, <classes> ).
        name : str
            Audio file name that will be included in the segments output.
            Choose a unique name, or the original audio
            file name, to distinguish multiple audio files. Default: None.
        speech_len : int
            Number of sample points. If given, the timing
            configuration is automatically derived from length of fs, length
            of speech and length of lpz. If None is given, make sure the
            timing parameters are correct, see time_stamps for reference!
            Default: None.

        Returns
        -------
        CTCSegmentationTask
            Task object that can be passed to
            ``CTCSegmentation.get_segments()`` in order to obtain alignments.
        """
        config = self.config
        # Update timing parameters, if needed
        if speech_len is not None:
            lpz_len = lpz.shape[0]
            timing_cfg = self.get_timing_config(speech_len, lpz_len)
            config.set(**timing_cfg)
        # `text` is needed in the form of a list.
        utt_ids, text = self._split_text(text)
        # Obtain utterance & label sequence from text
        if self.text_converter == "tokenize":
            # list of str --tokenize--> list of np.array
            token_list = [
                np.array(self._tokenizer.encode_as_ids(utt)) for utt in text
            ]
            # filter out any instances of the <unk> token
            unk = config.char_list.index("<unk>")
            token_list = [utt[utt != unk] for utt in token_list]
            ground_truth_mat, utt_begin_indices = prepare_token_list(
                config, token_list
            )
        else:
            assert self.text_converter == "classic"
            text_pieces = [
                "".join(self._tokenizer.encode_as_pieces(utt)) for utt in text
            ]
            # filter out any instances of the <unk> token
            text_pieces = [utt.replace("<unk>", "") for utt in text_pieces]
            ground_truth_mat, utt_begin_indices = prepare_text(
                config, text_pieces
            )
        task = CTCSegmentationTask(
            config=config,
            name=name,
            text=text,
            ground_truth_mat=ground_truth_mat,
            utt_begin_indices=utt_begin_indices,
            utt_ids=utt_ids,
            lpz=lpz,
        )
        return task

    @staticmethod
    def get_segments(task: CTCSegmentationTask):
        """Obtain segments for given utterance texts and CTC log posteriors.

        Args
        ----
        task : CTCSegmentationTask
            Task object that contains ground truth and
            CTC posterior probabilities.

        Returns
        -------
        dict
            Dictionary with alignments. Combine this with the task
            object to obtain a human-readable segments representation.
        """
        assert type(task) == CTCSegmentationTask
        assert task.config is not None
        config = task.config
        lpz = task.lpz
        ground_truth_mat = task.ground_truth_mat
        utt_begin_indices = task.utt_begin_indices
        text = task.text
        # Align using CTC segmentation
        timings, char_probs, state_list = ctc_segmentation(
            config, lpz, ground_truth_mat
        )
        # Obtain list of utterances with time intervals and confidence score
        segments = determine_utterance_segments(
            config, utt_begin_indices, char_probs, timings, text
        )
        # Store results
        result = {
            "name": task.name,
            "timings": timings,
            "char_probs": char_probs,
            "state_list": state_list,
            "segments": segments,
            "done": True,
        }
        return result

    def __call__(
        self,
        speech: Union[paddle.Tensor, np.ndarray, str, Path],
        text: Union[List[str], str],
        name: Optional[str] = None,
    ) -> CTCSegmentationTask:
        """Align utterances.

        Args
        ----
        speech : Union[paddle.Tensor, np.ndarray, str, Path]
            Audio file that can be given as path or as array.
        text : Union[List[str], str]
            List or multiline-string with utterance ground truths.
            The required formatting depends on the setting ``kaldi_style_text``.
        name : str
            Name of the file. Utterance names are derived from it.

        Returns
        -------
        CTCSegmentationTask
            Task object with segments. Apply str(·) or print(·) on it
            to obtain the segments list.
        """
        if isinstance(speech, str) or isinstance(speech, Path):
            speech = self.asr_model.load_audio(speech)
        # Get log CTC posterior probabilities
        lpz = self.get_lpz(speech)
        # Conflate text & lpz & config as a segmentation task object
        task = self.prepare_segmentation_task(text, lpz, name, speech.shape[0])
        # Apply CTC segmentation
        segments = self.get_segments(task)
        task.set(**segments)
        return task
