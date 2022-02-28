"""Library for Byte-pair-encoding (BPE) tokenization.
Authors
 * Abdelwahab Heba 2020
 * Loren Lugosch 2020
"""

import os.path
import paddle
import logging
import csv
import json
import sentencepiece as spm
from speechbrain.dataio.dataio import merge_char
from speechbrain.utils import edit_distance
from speechbrain.utils.distributed import run_on_main

logger = logging.getLogger(__name__)


class SentencePiece:
    """BPE class call the SentencePiece unsupervised text tokenizer from Google.
    Reference: https://github.com/google/sentencepiece
    SentencePiece lib is an unsupervised text tokenizer and detokenizer.
    It implements subword units like Byte-pair-encoding (BPE),
    Unigram language model and char/word tokenizer.
    Arguments
    ---------
    model_dir : str
        The directory where the model will be saved (or already stored).
    vocab_size : int, None, optional
        Vocab size for the chosen tokenizer type (BPE, Unigram).
        The vocab_size is optional for char, and mandatory for BPE & unigram
        tokenization.
    annotation_train : str
        Path of the annotation file which is used to learn the tokenizer. It
        can be in JSON or csv format.
    annotation_read : str
        The data entry which contains the word sequence in the annotation file.
    model_type : str
        (bpe, char, unigram).
        If "bpe", train unsupervised tokenization of piece of words. see:
        https://www.aclweb.org/anthology/P16-1162/
        If "word" take the vocabulary from the input text.
        If "unigram" do piece of word tokenization using unigram language
        model, see: https://arxiv.org/abs/1804.10959
    char_format_input : bool
        Whether the read entry contains characters format input.
        (default: False)
        (e.g., a p p l e _ i s _ g o o d)
    character_coverage : int
        Amount of characters covered by the model, good defaults
        are: 0.9995 for languages with a rich character set like Japanese or
        Chinese and 1.0 for other languages with small character set.
        (default: 1.0)
    user_defined_symbols : string
        String contained a list of symbols separated by a comma.
        User-defined symbols are handled as one piece in any context.
        (default: None)
    max_sentencepiece_length : int
        Maximum number of characters for the tokens. (default: 10)
    bos_id : int
        If -1 the bos_id = unk_id = 0. otherwise, bos_id = int. (default: -1)
    eos_id : int
        If -1 the bos_id = unk_id = 0. otherwise, bos_id = int. (default: -1)
    split_by_whitespace : bool
        If False, allow the sentencepiece to extract piece crossing multiple
        words. This feature is important for : Chinese/Japanese/Korean.
        (default: True)
    num_sequences : int
        If not none, use at most this many sequences to train the tokenizer
        (for large datasets). (default: None)
    annotation_list_to_check : list,
        List of the annotation file which is used for checking the accuracy of
        recovering words from the tokenizer.
    annotation_format : str
        The format of the annotation file. JSON or csv are the formats supported.
    add_dummy_prefix : bool
        If True the tokenizer adds dummy whitespace at the beginning of text. (default: True)
    Example
    -------
    >>> import paddle
    >>> dict_int2lab = {1: "HELLO", 2: "MORNING"}
    >>> model_dir = "tests/unittests/tokenizer_data/"
    >>> # Example with csv
    >>> annotation_train = "tests/unittests/tokenizer_data/dev-clean.csv"
    >>> annotation_read = "wrd"
    >>> model_type = "bpe"
    >>> bpe = SentencePiece(model_dir,100, annotation_train, annotation_read,
    ...                     model_type)
    >>> batch_seq = paddle.Tensor([[1, 2, 2, 1],[1, 2, 1, 0]])
    >>> batch_lens = paddle.Tensor([1.0, 0.75])
    >>> encoded_seq_ids, encoded_seq_pieces = bpe(
    ...     batch_seq, batch_lens, dict_int2lab, task="encode"
    ... )
    >>> # Example using JSON
    >>> annotation_train = "tests/unittests/tokenizer_data/dev-clean.json"
    >>> annotation_read = "wrd"
    >>> bpe = SentencePiece(model_dir,100, annotation_train, annotation_read,
    ...                     model_type, annotation_format = 'json')
    >>> encoded_seq_ids, encoded_seq_pieces = bpe(
    ...     batch_seq, batch_lens, dict_int2lab, task="encode"
    ... )
    """

    def __init__(
        self,
        model_dir,
        vocab_size,
        annotation_train=None,
        annotation_read=None,
        model_type="unigram",
        char_format_input=False,
        character_coverage=1.0,
        user_defined_symbols=None,
        max_sentencepiece_length=10,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
        split_by_whitespace=True,
        num_sequences=None,
        annotation_list_to_check=None,
        annotation_format="csv",
        add_dummy_prefix=True,
    ):
        if model_type not in ["unigram", "bpe", "char"]:
            raise ValueError("model_type must be one of : [unigram, bpe, char]")
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        if not isinstance(vocab_size, int):
            raise ValueError("vocab_size must be integer.")

        self.annotation_train = annotation_train
        self.annotation_read = annotation_read
        self.annotation_format = annotation_format

        if self.annotation_train is not None:
            ext = os.path.splitext(self.annotation_train)[1]
            self.text_file = self.annotation_train.replace(ext, ".txt")

        self.prefix_model_file = os.path.join(
            model_dir, str(vocab_size) + "_" + model_type
        )
        self.vocab_size = str(vocab_size)
        self.model_type = model_type
        self.char_format_input = char_format_input
        self.character_coverage = str(character_coverage)
        self.max_sentencepiece_length = str(max_sentencepiece_length)
        self.bos_id = str(bos_id)
        self.eos_id = str(eos_id)
        self.pad_id = str(pad_id)
        self.unk_id = str(unk_id)
        self.num_sequences = num_sequences
        self.split_by_whitespace = split_by_whitespace
        self.user_defined_symbols = user_defined_symbols
        self.add_dummy_prefix = str(add_dummy_prefix)

        if not os.path.isfile(self.prefix_model_file + ".model"):
            logger.info("Train tokenizer with type:" + self.model_type)
            if not os.path.isfile(self.text_file):
                if annotation_format == "csv":
                    run_on_main(self._csv2text)
                elif annotation_format == "json":
                    run_on_main(self._json2text)
                else:
                    raise ValueError(
                        "Annotation format not supported. Supported formats are csv and json. Got "
                        + annotation_format
                    )
            run_on_main(self._train_BPE)
        else:
            logger.info("Tokenizer is already trained.")

        logger.info("==== Loading Tokenizer ===")
        logger.info("Tokenizer path: " + self.prefix_model_file + ".model")
        logger.info("Tokenizer vocab_size: " + str(self.vocab_size))
        logger.info("Tokenizer type: " + self.model_type)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.prefix_model_file + ".model")

        if annotation_list_to_check is not None:
            run_on_main(
                self._check_coverage_from_bpe,
                kwargs={"list_annotation_files": annotation_list_to_check},
            )

    def _csv2text(self):
        """Read CSV file and convert specific data entries into text file.
        """
        if not os.path.isfile(os.path.abspath(self.annotation_train)):
            raise ValueError(
                self.annotation_train
                + " is not a file. please provide annotation file for training."
            )
        logger.info(
            "Extract "
            + self.annotation_read
            + " sequences from:"
            + self.annotation_train
        )
        annotation_file = open(self.annotation_train, "r")
        reader = csv.reader(annotation_file)
        headers = next(reader, None)
        if self.annotation_read not in headers:
            raise ValueError(
                self.annotation_read + " must exist in:" + self.annotation_train
            )
        index_label = headers.index(self.annotation_read)
        text_file = open(self.text_file, "w+")
        row_idx = 0
        for row in reader:
            if self.num_sequences is not None and row_idx > self.num_sequences:
                print(
                    "Using %d sequences to train the tokenizer."
                    % self.num_sequences
                )
                break
            row_idx += 1
            sent = row[index_label]
            if self.char_format_input:
                (sent,) = merge_char([sent.split()])
                sent = " ".join(sent)
            text_file.write(sent + "\n")
        text_file.close()
        annotation_file.close()
        logger.info("Text file created at: " + self.text_file)

    def _json2text(self):
        """Read JSON file and convert specific data entries into text file.
        """
        if not os.path.isfile(os.path.abspath(self.annotation_train)):
            raise ValueError(
                self.annotation_train
                + " is not a file. please provide annotation file for training."
            )
        logger.info(
            "Extract "
            + self.annotation_read
            + " sequences from:"
            + self.annotation_train
        )

        # Read JSON
        with open(self.annotation_train, "r") as f:
            out_json = json.load(f)

        # Save text file
        text_file = open(self.text_file, "w+")
        row_idx = 0

        for snt_id in out_json.keys():
            if self.num_sequences is not None and row_idx > self.num_sequences:
                print(
                    "Using %d sequences to train the tokenizer."
                    % self.num_sequences
                )
                break
            row_idx += 1
            sent = out_json[snt_id][self.annotation_read]
            if self.char_format_input:
                (sent,) = merge_char([sent.split()])
                sent = " ".join(sent)

            text_file.write(sent + "\n")
        text_file.close()

        logger.info("Text file created at: " + self.text_file)

    def _train_BPE(self):
        """Train tokenizer with unsupervised techniques (BPE, Unigram) using
        SentencePiece Library. If you use "char" mode, the SentencePiece
        creates a char dict so the vocab_size attribute is not needed.
        """
        query = (
            "--input="
            + self.text_file
            + " --model_prefix="
            + self.prefix_model_file
            + " --model_type="
            + self.model_type
            + " --bos_id="
            + self.bos_id
            + " --eos_id="
            + self.eos_id
            + " --pad_id="
            + self.pad_id
            + " --unk_id="
            + self.unk_id
            + " --max_sentencepiece_length="
            + self.max_sentencepiece_length
            + " --character_coverage="
            + self.character_coverage
            + " --add_dummy_prefix="
            + self.add_dummy_prefix
        )
        if self.model_type not in ["char"]:
            # include vocab_size
            query += " --vocab_size=" + str(self.vocab_size)
        if self.user_defined_symbols is not None:
            query += " --user_defined_symbols=" + self.user_defined_symbols
        if not self.split_by_whitespace:
            query += " --split_by_whitespace=false"
        # Train tokenizer
        spm.SentencePieceTrainer.train(query)

    def _check_coverage_from_bpe(self, list_annotation_files=[]):
        """Logging the accuracy of the BPE model to recover words from the training text.
        Arguments
        ---------
        annotation_list_to_check : list,
            List of the annotation file which is used for checking the accuracy of recovering words from the tokenizer.
        """
        for annotation_file in list_annotation_files:
            if os.path.isfile(os.path.abspath(annotation_file)):
                logger.info(
                    "==== Accuracy checking for recovering text from tokenizer ==="
                )
                # csv reading
                if self.annotation_format == "csv":
                    fannotation_file = open(annotation_file, "r")
                    reader = csv.reader(fannotation_file)
                    headers = next(reader, None)
                    if self.annotation_read not in headers:
                        raise ValueError(
                            self.annotation_read
                            + " must exist in:"
                            + annotation_file
                        )
                    index_label = headers.index(self.annotation_read)
                # json reading
                else:
                    with open(self.annotation_train, "r") as f:
                        reader = json.load(f)
                        index_label = self.annotation_read

                wrong_recover_list = []
                for row in reader:
                    if self.annotation_format == "csv":
                        row = row[index_label]
                    else:
                        row = reader[row][index_label]
                    if self.char_format_input:
                        (row,) = merge_char([row.split()])
                        row = " ".join(row)
                    row = row.split("\n")[0]
                    encoded_id = self.sp.encode_as_ids(row)
                    decode_text = self.sp.decode_ids(encoded_id)
                    (details,) = edit_distance.wer_details_for_batch(
                        ["utt1"],
                        [row.split(" ")],
                        [decode_text.split(" ")],
                        compute_alignments=True,
                    )
                    if details["WER"] > 0:
                        for align in details["alignment"]:
                            if align[0] != "=" and align[1] is not None:
                                if align[1] not in wrong_recover_list:
                                    wrong_recover_list.append(align[1])
                if self.annotation_format == "csv":
                    fannotation_file.close()
                logger.info("recover words from: " + annotation_file)
                if len(wrong_recover_list) > 0:
                    logger.warn(
                        "Wrong recover words: " + str(len(wrong_recover_list))
                    )
                    logger.warn(
                        "Tokenizer vocab size: " + str(self.sp.vocab_size())
                    )
                    logger.warn(
                        "accuracy recovering words: "
                        + str(
                            1
                            - float(len(wrong_recover_list))
                            / self.sp.vocab_size()
                        )
                    )
                else:
                    logger.info("Wrong recover words: 0")
                    logger.warning("accuracy recovering words: " + str(1.0))
            else:
                logger.info(
                    "No accuracy recover checking for" + annotation_file
                )

    def __call__(
        self, batch, batch_lens=None, ind2lab=None, task="encode",
    ):
        """This __call__ function implements the tokenizer encoder and decoder
        (restoring the string of word) for BPE, Regularized BPE (with unigram),
        and char (speechbrain/nnet/RNN.py).
        Arguments
        ----------
        batch : tensor.IntTensor or list
            List if ( batch_lens = None and task = "decode_from_list")
            Contains the original labels. Shape: [batch_size, max_length]
        batch_lens : tensor.LongTensor
            Containing the relative length of each label sequences. Must be 1D
            tensor of shape: [batch_size]. (default: None)
        ind2lab : dict
            Dictionary which maps the index from label sequences
            (batch tensor) to string label.
        task : str
            ("encode", "decode", "decode_from_list)
            "encode": convert the batch tensor into sequence of tokens.
                the output contain a list of (tokens_seq, tokens_lens)
            "decode": convert a tensor of tokens to a list of word sequences.
            "decode_from_list": convert a list of token sequences to a list
                of word sequences.
        """
        if task == "encode" and ind2lab is None:
            raise ValueError("Tokenizer encoder must have the ind2lab function")

        if task == "encode":
            # Convert list of words/chars to bpe ids
            bpe = []
            max_bpe_len = 0
            batch_lens = (batch_lens * batch.shape[1]).round().int()
            for i, utt_seq in enumerate(batch):
                tokens = [
                    ind2lab[int(index)] for index in utt_seq[: batch_lens[i]]
                ]
                if self.char_format_input:
                    (words_list,) = merge_char([tokens])
                    sent = " ".join(words_list)
                else:
                    sent = " ".join(tokens)
                bpe_encode = self.sp.encode_as_ids(sent)
                bpe.append(bpe_encode)
                # save the longest bpe sequence
                # it help to compute the relative length of each utterance
                if len(bpe_encode) > max_bpe_len:
                    max_bpe_len = len(bpe_encode)
            # Create bpe tensor
            bpe_tensor = torch.zeros(
                (batch.shape[0], max_bpe_len), device=batch.device
            )
            bpe_lens = torch.zeros((batch.shape[0]), device=batch.device)
            for i, bpe_utt in enumerate(bpe):
                bpe_tensor[i, : len(bpe_utt)] = paddle.Tensor(bpe_utt)
                bpe_lens[i] = len(bpe_utt) / max_bpe_len
            return bpe_tensor, bpe_lens
        elif task == "decode_from_list":
            # From list of hyps (not padded outputs)
            # do decoding
            return [self.sp.decode_ids(utt_seq).split(" ") for utt_seq in batch]
        elif task == "decode":
            # From a batch tensor and a length tensor
            # find the absolute batch lengths and do decoding
            batch_lens = (batch_lens * batch.shape[1]).round().int()
            return [
                self.sp.decode_ids(
                    utt_seq[: batch_lens[i]].int().tolist()
                ).split(" ")
                for i, utt_seq in enumerate(batch)
            ]
