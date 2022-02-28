"""Encoding categorical data as integers

Authors
  * Samuele Cornell 2020
  * Aku Rouhe 2020
"""
import ast
import paddle
import collections
import itertools
import logging
import speechbrain as sb
from speechbrain.utils.checkpoints import (
    mark_as_saver,
    mark_as_loader,
    register_checkpoint_hooks,
)

logger = logging.getLogger(__name__)

# NOTE: Changing these does NOT change the defaults in the classes.
# Consider these read-only.
DEFAULT_UNK = "<unk>"
DEFAULT_BOS = "<bos>"
DEFAULT_EOS = "<eos>"
DEFAULT_BLANK = "<blank>"


@register_checkpoint_hooks
class CategoricalEncoder:
    """Encode labels of a discrete set.

    Used for encoding, e.g., speaker identities in speaker recognition.
    Given a collection of hashables (e.g a strings) it encodes
    every unique item to an integer value: ["spk0", "spk1"] --> [0, 1]
    Internally the correspondence between each label to its index is handled by
    two dictionaries: lab2ind and ind2lab.

    The label integer encoding can be generated automatically from a SpeechBrain
    DynamicItemDataset by specifying the desired entry (e.g., spkid) in the annotation
    and calling update_from_didataset method:

    >>> from speechbrain.dataio.encoder import CategoricalEncoder
    >>> from speechbrain.dataio.dataset import DynamicItemDataset
    >>> dataset = {"ex_{}".format(x) : {"spkid" : "spk{}".format(x)} for x in range(20)}
    >>> dataset = DynamicItemDataset(dataset)
    >>> encoder = CategoricalEncoder()
    >>> encoder.update_from_didataset(dataset, "spkid")
    >>> assert len(encoder) == len(dataset) # different speaker for each utterance

    However can also be updated from an iterable:

    >>> from speechbrain.dataio.encoder import CategoricalEncoder
    >>> from speechbrain.dataio.dataset import DynamicItemDataset
    >>> dataset = ["spk{}".format(x) for x in range(20)]
    >>> encoder = CategoricalEncoder()
    >>> encoder.update_from_iterable(dataset)
    >>> assert len(encoder) == len(dataset)

    Note
    ----
    In both methods it can be specified it the single element in the iterable
    or in the dataset should be treated as a sequence or not (default False).
    If it is a sequence each element in the sequence will be encoded.


    >>> from speechbrain.dataio.encoder import CategoricalEncoder
    >>> from speechbrain.dataio.dataset import DynamicItemDataset
    >>> dataset = [[x+1, x+2] for x in range(20)]
    >>> encoder = CategoricalEncoder()
    >>> encoder.update_from_iterable(dataset, sequence_input=True)
    >>> assert len(encoder) == 21 # there are only 21 unique elements 1-21

    This class offers 4 different methods to explicitly add a label in the internal
    dicts: add_label, ensure_label, insert_label, enforce_label.
    add_label and insert_label will raise an error if it is already present in the
    internal dicts. insert_label, enforce_label allow also to specify the integer value
    to which the desired label is encoded.

    Encoding can be performed using 4 different methods:
    encode_label, encode_sequence, encode_label_torch and encode_sequence_torch.
    encode_label operate on single labels and simply returns the corresponding
    integer encoding:

    >>> from speechbrain.dataio.encoder import CategoricalEncoder
    >>> from speechbrain.dataio.dataset import DynamicItemDataset
    >>> dataset = ["spk{}".format(x) for x in range(20)]
    >>> encoder.update_from_iterable(dataset)
    >>>
    22
    >>>
    encode_sequence on sequences of labels:
    >>> encoder.encode_sequence(["spk1", "spk19"])
    [22, 40]
    >>>
    encode_label_torch and encode_sequence_torch return paddle tensors
    >>> encoder.encode_sequence_torch(["spk1", "spk19"])
    tensor([22, 40])
    >>>
    Decoding can be performed using decode_torch and decode_ndim methods.
    >>> encoded = encoder.encode_sequence_torch(["spk1", "spk19"])
    >>> encoder.decode_torch(encoded)
    ['spk1', 'spk19']
    >>>
    decode_ndim is used for multidimensional list or pytorch tensors
    >>> encoded = encoded.unsqueeze(0).repeat(3, 1)
    >>> encoder.decode_torch(encoded)
    [['spk1', 'spk19'], ['spk1', 'spk19'], ['spk1', 'spk19']]
    >>>

    In some applications, it can happen that during testing a label which has not
    been encountered during training is encountered. To handle this out-of-vocabulary
    problem add_unk can be used. Every out-of-vocab label is mapped to this special
    <unk> label and its corresponding integer encoding.

    >>> import paddle
    >>> try:
    ...     encoder.encode_label("spk42")
    ... except KeyError:
    ...        print("spk42 is not in the encoder this raises an error!")
    spk42 is not in the encoder this raises an error!
    >>> encoder.add_unk()
    41
    >>> encoder.encode_label("spk42")
    41
    >>>
    returns the <unk> encoding

    This class offers also methods to save and load the internal mappings between
    labels and tokens using: save and load methods as well as load_or_create.
    """

    VALUE_SEPARATOR = " => "
    EXTRAS_SEPARATOR = "================\n"

    def __init__(self, starting_index=0, **special_labels):
        self.lab2ind = {}
        self.ind2lab = {}
        self.starting_index = starting_index
        # NOTE: unk_label is not necessarily set at all!
        # This is because None is a suitable value for unk.
        # So the test is: hasattr(self, "unk_label")
        # rather than self.unk_label is not None
        self.handle_special_labels(special_labels)

    def handle_special_labels(self, special_labels):
        if "unk_label" in special_labels:
            self.add_unk(special_labels["unk_label"])

    def __len__(self):
        return len(self.lab2ind)

    @classmethod
    def from_saved(cls, path):
        """Recreate a previously saved encoder directly"""
        obj = cls()
        obj.load(path)
        return obj

    def update_from_iterable(self, iterable, sequence_input=False):
        """Update from iterator

        Arguments
        ---------
        iterable : iterable
            Input sequence on which to operate.
        sequence_input : bool
            Whether iterable yields sequences of labels or individual labels
            directly. (default False)
        """
        if sequence_input:
            label_iterator = itertools.chain.from_iterable(iterable)
        else:
            label_iterator = iter(iterable)
        for label in label_iterator:
            self.ensure_label(label)

    def update_from_didataset(
        self, didataset, output_key, sequence_input=False
    ):
        """Update from DynamicItemDataset.

        Arguments
        ---------
        didataset : DynamicItemDataset
            Dataset on which to operate.
        output_key : str
            Key in the dataset (in data or a dynamic item) to encode.
        sequence_input : bool
            Whether the data yielded with the specified key consists of
            sequences of labels or individual labels directly.
        """
        with didataset.output_keys_as([output_key]):
            self.update_from_iterable(
                (data_point[output_key] for data_point in didataset),
                sequence_input=sequence_input,
            )

    def limited_labelset_from_iterable(
        self, iterable, sequence_input=False, n_most_common=None, min_count=1
    ):
        """Produce label mapping from iterable based on label counts

        Used to limit label set size.

        Arguments
        ---------
        iterable : iterable
            Input sequence on which to operate.
        sequence_input : bool
            Whether iterable yields sequences of labels or individual labels
            directly. False by default.
        n_most_common : int, None
            Take at most this many labels as the label set, keeping the most
            common ones. If None (as by default), take all.
        min_count : int
            Don't take labels if they appear less than this many times.

        Returns
        -------
        collections.Counter
            The counts of the different labels (unfiltered).
        """
        if self.lab2ind:
            clsname = self.__class__.__name__
            logger.info(
                f"Limited_labelset_from_iterable called, "
                f"but {clsname} is not empty. "
                "The new labels will be added, i.e. won't overwrite. "
                "This is normal if there is e.g. an unk label already."
            )
        if sequence_input:
            label_iterator = itertools.chain.from_iterable(iterable)
        else:
            label_iterator = iter(iterable)
        counts = collections.Counter(label_iterator)
        for label, count in counts.most_common(n_most_common):
            if count < min_count:
                # .most_common() produces counts in descending order,
                # so no more labels can be found
                break
            self.add_label(label)
        return counts

    def load_or_create(
        self,
        path,
        from_iterables=[],
        from_didatasets=[],
        sequence_input=False,
        output_key=None,
        special_labels={},
    ):
        """Convenient syntax for creating the encoder conditionally

        This pattern would be repeated in so many experiments that
        we decided to add a convenient shortcut for it here. The
        current version is multi-gpu (DDP) safe.
        """
        try:
            if sb.utils.distributed.if_main_process():
                if not self.load_if_possible(path):
                    for iterable in from_iterables:
                        self.update_from_iterable(iterable, sequence_input)
                    for didataset in from_didatasets:
                        if output_key is None:
                            raise ValueError(
                                "Provide an output_key for "
                                "DynamicItemDataset"
                            )
                        # 从dataset中的output_key中获取label的数据
                        self.update_from_didataset(
                            didataset, output_key, sequence_input
                        )
                    self.handle_special_labels(special_labels)
                    self.save(path)
        finally:
            sb.utils.distributed.ddp_barrier()
            self.load(path)

    def add_label(self, label):
        """Add new label to the encoder, at the next free position.

        Arguments
        ---------
        label : hashable
            Most often labels are str, but anything that can act as dict key is
            supported. Note that default save/load only supports Python
            literals.

        Returns
        -------
        int
            The index that was used to encode this label.
        """
        if label in self.lab2ind:
            clsname = self.__class__.__name__
            raise KeyError(f"Label already present in {clsname}")
        index = self._next_index()
        self.lab2ind[label] = index
        self.ind2lab[index] = label
        return index

    def ensure_label(self, label):
        """Add a label if it is not already present.

        Arguments
        ---------
        label : hashable
            Most often labels are str, but anything that can act as dict key is
            supported. Note that default save/load only supports Python
            literals.

        Returns
        -------
        int
            The index that was used to encode this label.
        """
        if label in self.lab2ind:
            return self.lab2ind[label]
        else:
            return self.add_label(label)

    def insert_label(self, label, index):
        """Add a new label, forcing its index to a specific value.

        If a label already has the specified index, it is moved to the end
        of the mapping.

        Arguments
        ---------
        label : hashable
            Most often labels are str, but anything that can act as dict key is
            supported. Note that default save/load only supports Python
            literals.
        index : int
            The specific index to use.
        """
        if label in self.lab2ind:
            clsname = self.__class__.__name__
            raise KeyError(f"Label already present in {clsname}")
        else:
            self.enforce_label(label, index)

    def enforce_label(self, label, index):
        """Make sure label is present and encoded to a particular index.

        If the label is present but encoded to some other index, it is
        moved to the given index.

        If there is already another label at the
        given index, that label is moved to the next free position.
        """
        index = int(index)
        if label in self.lab2ind:
            if index == self.lab2ind[label]:
                return
            else:
                # Delete old index mapping. Everything else gets overwritten.
                del self.ind2lab[self.lab2ind[label]]
        # Move other label out of the way:
        if index in self.ind2lab:
            saved_label = self.ind2lab[index]
            moving_other = True
        else:
            moving_other = False
        # Ready to push the new index.
        self.lab2ind[label] = index
        self.ind2lab[index] = label
        # And finally put the moved index in new spot.
        if moving_other:
            logger.info(
                f"Moving label {repr(saved_label)} from index "
                f"{index}, because {repr(label)} was put at its place."
            )
            new_index = self._next_index()
            self.lab2ind[saved_label] = new_index
            self.ind2lab[new_index] = saved_label

    def add_unk(self, unk_label=DEFAULT_UNK):
        """Add label for unknown tokens (out-of-vocab).

        When asked to encode unknown labels, they can be mapped to this.

        Arguments
        ---------
        label : hashable, optional
            Most often labels are str, but anything that can act as dict key is
            supported. Note that default save/load only supports Python
            literals. Default: <unk>. This can be None, as well!

        Returns
        -------
        int
            The index that was used to encode this.
        """
        self.unk_label = unk_label
        return self.add_label(unk_label)

    def _next_index(self):
        """The index to use for the next new label"""
        index = self.starting_index
        while index in self.ind2lab:
            index += 1
        return index

    def is_continuous(self):
        """Check that the set of indices doesn't have gaps

        For example:
        If starting index = 1
        Continuous: [1,2,3,4]
        Continuous: [0,1,2]
        Non-continuous: [2,3,4]
        Non-continuous: [1,2,4]

        Returns
        -------
        bool
            True if continuous.
        """
        # Because of Python indexing this also handles the special cases
        # of 0 or 1 labels.
        indices = sorted(self.ind2lab.keys())
        return self.starting_index in indices and all(
            j - i == 1 for i, j in zip(indices[:-1], indices[1:])
        )

    def encode_label(self, label, allow_unk=True):
        """Encode label to int

        Arguments
        ---------
        label : hashable
            Label to encode, must exist in the mapping.
        allow_unk : bool
            If given, that label is not in the label set
            AND unk_label has been added with add_unk(),
            allows encoding to unk_label's index.

        Returns
        -------
        int
            Corresponding encoded int value.
        """
        try:
            return self.lab2ind[label]
        except KeyError:
            if hasattr(self, "unk_label") and allow_unk:
                return self.lab2ind[self.unk_label]
            elif hasattr(self, "unk_label") and not allow_unk:
                raise KeyError(
                    f"Unknown label {label}, and explicitly "
                    "disallowed the use of the existing unk-label"
                )
            elif not hasattr(self, "unk_label") and allow_unk:
                raise KeyError(
                    f"Cannot encode unknown label {label}. "
                    "You have not called add_unk() to add a special "
                    "unk-label for unknown labels."
                )
            else:
                raise KeyError(
                    f"Couldn't and wouldn't encode unknown label " f"{label}."
                )

    def encode_label_torch(self, label, allow_unk=True):
        """Encode label to paddle.LongTensor.

        Arguments
        ---------
        label : hashable
            Label to encode, must exist in the mapping.

        Returns
        -------
        paddle.LongTensor
            Corresponding encoded int value.
            Tensor shape [1].
        """
        return paddle.LongTensor([self.encode_label(label, allow_unk)])

    def encode_sequence(self, sequence, allow_unk=True):
        """Encode a sequence of labels to list

        Arguments
        ---------
        x : iterable
            Labels to encode, must exist in the mapping.

        Returns
        -------
        list
            Corresponding integer labels.
        """
        return [self.encode_label(label, allow_unk) for label in sequence]

    def encode_sequence_paddle(self, sequence, allow_unk=True):
        """Encode a sequence of labels to paddle.LongTensor

        Arguments
        ---------
        x : iterable
            Labels to encode, must exist in the mapping.

        Returns
        -------
        paddle.LongTensor
            Corresponding integer labels.
            Tensor shape [len(sequence)].
        """
        return paddle.to_tensor(
            [self.encode_label(label, allow_unk) for label in sequence]
            , dtype="int64"
        )

    def decode_paddle(self, x):
        """Decodes an arbitrarily nested paddle.Tensor to a list of labels.

        Provided separately because Torch provides clearer introspection,
        and so doesn't require try-except.

        Arguments
        ---------
        x : paddle.Tensor
            Torch tensor of some integer dtype (Long, int) and any shape to
            decode.

        Returns
        -------
        list
            list of original labels
        """
        decoded = []
        # Recursively operates on the different dimensions.
        if x.ndim == 1:  # Last dimension!
            for element in x:
                decoded.append(self.ind2lab[int(element)])
        else:
            for subtensor in x:
                decoded.append(self.decode_paddle(subtensor))
        return decoded

    def decode_ndim(self, x):
        """Decodes an arbitrarily nested iterable to a list of labels.

        This works for essentially any pythonic iterable (including paddle), and
        also single elements.

        Arguments
        ---------
        x : Any
            Python list or other iterable or paddle.Tensor or a single integer element

        Returns
        -------
        list, Any
            ndim list of original labels, or if input was single element,
            output will be, too.
        """
        # Recursively operates on the different dimensions.
        try:
            decoded = []
            if isinstance(x, paddle.Tensor):
                # 如果使用paddle.Tensor进行迭代的话，有bug
                x = x.tolist()
            for subtensor in x:
                print("subtensor: {}".format(subtensor))
                decoded.append(self.decode_ndim(subtensor))
            return decoded
        except TypeError:  # Not an iterable, bottom level!
            return self.ind2lab[int(x)]

    @mark_as_saver
    def save(self, path):
        """Save the categorical encoding for later use and recovery

        Saving uses a Python literal format, which supports things like
        tuple labels, but is considered safe to load (unlike e.g. pickle).

        Arguments
        ---------
        path : str, Path
            Where to save. Will overwrite.
        """
        extras = self._get_extras()
        self._save_literal(path, self.lab2ind, extras)

    def load(self, path):
        """Loads from the given path.

        CategoricalEncoder uses a Python literal format, which supports things
        like tuple labels, but is considered safe to load (unlike e.g. pickle).

        Arguments
        ---------
        path : str, Path
            Where to load from.
        """
        if self.lab2ind:
            clsname = self.__class__.__name__
            logger.info(
                f"Load called, but {clsname} is not empty. "
                "Loaded data will overwrite everything. "
                "This is normal if there is e.g. an unk label defined at init."
            )
        lab2ind, ind2lab, extras = self._load_literal(path)
        self.lab2ind = lab2ind
        self.ind2lab = ind2lab
        self._set_extras(extras)
        # If we're here, load was a success!
        logger.debug(f"Loaded categorical encoding from {path}")

    @mark_as_loader
    def load_if_possible(self, path, end_of_epoch=False, device=None):
        """Loads if possible, returns a bool indicating if loaded or not.

        Arguments
        ---------
        path : str, Path
            Where to load from.

        Returns
        -------
        bool :
            If load was successful.

        Example
        -------
        >>> encoding_file = getfixture('tmpdir') / "encoding.txt"
        >>> encoder = CategoricalEncoder()
        >>> # The idea is in an experiment script to have something like this:
        >>> if not encoder.load_if_possible(encoding_file):
        ...     encoder.update_from_iterable("abcd")
        ...     encoder.save(encoding_file)
        >>> # So the first time you run the experiment, the encoding is created.
        >>> # However, later, the encoding exists:
        >>> encoder = CategoricalEncoder()
        >>> if not encoder.load_if_possible(encoding_file):
        ...     assert False  # We won't get here!
        >>> encoder.decode_ndim(range(4))
        ['a', 'b', 'c', 'd']
        """
        del end_of_epoch  # Unused here.
        del device  # Unused here.

        try:
            self.load(path)
        except FileNotFoundError:
            logger.debug(
                f"Would load categorical encoding from {path}, "
                "but file doesn't exist yet."
            )
            return False
        except (ValueError, SyntaxError):
            logger.debug(
                f"Would load categorical encoding from {path}, "
                "and file existed but seems to be corrupted or otherwise couldn't load."
            )
            return False
        return True  # If here, all good

    def _get_extras(self):
        """Override this to provide any additional things to save

        Call super()._get_extras() to get the base extras
        """
        extras = {"starting_index": self.starting_index}
        if hasattr(self, "unk_label"):
            extras["unk_label"] = self.unk_label
        return extras

    def _set_extras(self, extras):
        """Override this to e.g. load any extras needed

        Call super()._set_extras(extras) to set the base extras
        """
        if "unk_label" in extras:
            self.unk_label = extras["unk_label"]
        self.starting_index = extras["starting_index"]

    @staticmethod
    def _save_literal(path, lab2ind, extras):
        """Save which is compatible with _load_literal"""
        with open(path, "w") as f:
            for label, ind in lab2ind.items():
                f.write(
                    repr(label)
                    + CategoricalEncoder.VALUE_SEPARATOR
                    + str(ind)
                    + "\n"
                )
            f.write(CategoricalEncoder.EXTRAS_SEPARATOR)
            for key, value in extras.items():
                f.write(
                    repr(key)
                    + CategoricalEncoder.VALUE_SEPARATOR
                    + repr(value)
                    + "\n"
                )
            f.flush()

    @staticmethod
    def _load_literal(path):
        """Load which supports Python literals as keys.

        This is considered safe for user input, as well (unlike e.g. pickle).
        """
        lab2ind = {}
        ind2lab = {}
        extras = {}
        with open(path) as f:
            # Load the label to index mapping (until EXTRAS_SEPARATOR)
            for line in f:
                if line == CategoricalEncoder.EXTRAS_SEPARATOR:
                    break
                literal, ind = line.strip().split(
                    CategoricalEncoder.VALUE_SEPARATOR, maxsplit=1
                )
                ind = int(ind)
                label = ast.literal_eval(literal)
                lab2ind[label] = ind
                ind2lab[ind] = label
            # Load the extras:
            for line in f:
                literal_key, literal_value = line.strip().split(
                    CategoricalEncoder.VALUE_SEPARATOR, maxsplit=1
                )
                key = ast.literal_eval(literal_key)
                value = ast.literal_eval(literal_value)
                extras[key] = value
        return lab2ind, ind2lab, extras


class TextEncoder(CategoricalEncoder):
    """CategoricalEncoder subclass which offers specific methods for encoding text and handle
    special tokens for training of sequence to sequence models.
    In detail, aside special <unk> token already present in CategoricalEncoder
    for handling out-of-vocab tokens here special methods to handle
    <bos> beginning of sequence and <eos> tokens are defined.

    Note: update_from_iterable and update_from_didataset here have as default
    sequence_input=True because it is assumed that this encoder is used on
    iterables of strings: e.g.

    >>> from speechbrain.dataio.encoder import TextEncoder
    >>> dataset = [["encode", "this", "textencoder"], ["foo", "bar"]]
    >>> encoder = TextEncoder()
    >>> encoder.update_from_iterable(dataset)
    >>> encoder.encode_label("this")
    1
    >>> encoder.add_unk()
    5
    >>> encoder.encode_sequence(["this", "out-of-vocab"])
    [1, 5]
    >>>

    Two methods can be used to add <bos> and <eos> to the internal dicts:
    insert_bos_eos, add_bos_eos.

    >>> encoder.add_bos_eos()
    >>> encoder.lab2ind[encoder.eos_label]
    7
    >>>
    add_bos_eos adds the special tokens at the end of the dict indexes
    >>> encoder = TextEncoder()
    >>> encoder.update_from_iterable(dataset)
    >>> encoder.insert_bos_eos(bos_index=0, eos_index=1)
    >>> encoder.lab2ind[encoder.eos_label]
    1
    >>>
    insert_bos_eos allows to specify whose index will correspond to each of them.
    Note that you can also specify the same integer encoding for both.

    Four methods can be used to prepend <bos> and append <eos>.
    prepend_bos_label and append_eos_label add respectively the <bos> and <eos>
    string tokens to the input sequence

    >>> words = ["foo", "bar"]
    >>> encoder.prepend_bos_label(words)
    ['<bos>', 'foo', 'bar']
    >>> encoder.append_eos_label(words)
    ['foo', 'bar', '<eos>']

    prepend_bos_index and append_eos_index add respectively the <bos> and <eos>
    indexes to the input encoded sequence.

    >>> words = ["foo", "bar"]
    >>> encoded = encoder.encode_sequence(words)
    >>> encoder.prepend_bos_index(encoded)
    [0, 3, 4]
    >>> encoder.append_eos_index(encoded)
    [3, 4, 1]

    """

    def handle_special_labels(self, special_labels):
        super().handle_special_labels(special_labels)
        # NOTE: bos_label and eos_label are not necessarily set at all!
        # This is because None is a suitable value.
        # So the test is: hasattr(self, "bos_label")
        # rather than self.bos_label is not None
        # Same thing with unk, see base class.
        if "bos_label" in special_labels and "eos_label" in special_labels:
            self.insert_bos_eos(
                bos_label="<bos>",
                eos_label="<eos>",
                bos_index=special_labels["bos_label"],
                eos_index=special_labels["eos_label"],
            )
        elif "bos_label" in special_labels or "eos_label" in special_labels:
            raise TypeError("Only BOS or EOS specified. Need both for init.")

    def update_from_iterable(self, iterable, sequence_input=True):
        """Change default for sequence_input to True."""
        return super().update_from_iterable(iterable, sequence_input)

    def update_from_didataset(self, didataset, output_key, sequence_input=True):
        """Change default for sequence_input to True."""
        return super().update_from_didataset(
            didataset, output_key, sequence_input
        )

    def limited_labelset_from_iterable(
        self, iterable, sequence_input=True, n_most_common=None, min_count=1
    ):
        """Change default for sequence_input to True."""
        return super().limited_labelset_from_iterable(
            iterable, sequence_input=True, n_most_common=None, min_count=1
        )

    def add_bos_eos(
        self, bos_label=DEFAULT_BOS, eos_label=DEFAULT_EOS,
    ):
        """Add sentence boundary markers in the label set.

        If the beginning-of-sentence and end-of-sentence markers
        are the same, will just use one sentence-boundary label.

        This method adds to the end of the index, rather than at the beginning,
        like insert_bos_eos.

        Arguments
        ---------
        bos_label : hashable
            Beginning-of-sentence label, any label.
        eos_label : hashable
            End-of-sentence label, any label. If set to the same label as
            bos_label, will just use one sentence-boundary label.
        """
        if bos_label == eos_label:
            logger.debug(
                "BOS and EOS labels are the same so using just one sentence "
                "boundary label"
            )
            self.add_label(bos_label)
        else:
            self.add_label(bos_label)
            self.add_label(eos_label)
        self.bos_label = bos_label
        self.eos_label = eos_label

    def insert_bos_eos(
        self,
        bos_label=DEFAULT_BOS,
        eos_label=DEFAULT_EOS,
        bos_index=0,
        eos_index=None,
    ):
        """Insert sentence boundary markers in the label set.

        If the beginning-of-sentence and end-of-sentence markers
        are the same, will just use one sentence-boundary label.

        Arguments
        ---------
        bos_label : hashable
            Beginning-of-sentence label, any label
        eos_label : hashable
            End-of-sentence label, any label. If set to the same label as
            bos_label, will just use one sentence-boundary label.
        bos_index : int
            Where to insert bos_label. eos_index = bos_index + 1
        bos_index : optional, int
            Where to insert eos_label. Default: eos_index = bos_index + 1
        """
        if bos_label == eos_label:
            logger.debug(
                "BOS and EOS labels are the same so using just one sentence "
                "boundary label"
            )
            self.insert_label(bos_label, bos_index)
        else:
            self.insert_label(bos_label, bos_index)
            if eos_index is None:
                logger.debug("EOS label not specified, using BOS label + 1")
                self.insert_label(eos_label, bos_index + 1)
            else:
                self.insert_label(eos_label, eos_index)
        self.bos_label = bos_label
        self.eos_label = eos_label

    def get_bos_index(self):
        """Returns the index to which blank encodes"""
        if not hasattr(self, "bos_label"):
            raise RuntimeError("BOS label is not set!")
        return self.encode_label(self.bos_label)

    def get_eos_index(self):
        """Returns the index to which blank encodes"""
        if not hasattr(self, "eos_label"):
            raise RuntimeError("EOS label is not set!")
        return self.encode_label(self.eos_label)

    def prepend_bos_label(self, x):
        """Returns a list version of x, with BOS prepended"""
        if not hasattr(self, "bos_label"):
            raise KeyError("BOS label has not been added to label set!")
        return [self.bos_label] + list(x)

    def prepend_bos_index(self, x):
        """Returns a list version of x, with BOS index prepended.
        If the input is a tensor, a tensor is returned."""
        if not hasattr(self, "bos_label"):
            raise KeyError("BOS label has not been added to label set!")
        if paddle.is_tensor(x):
            bos_ind = paddle.Tensor([self.lab2ind[self.bos_label]])
            return paddle.cat([bos_ind, x])
        return [self.lab2ind[self.bos_label]] + list(x)

    def append_eos_label(self, x):
        """Returns a list version of x, with EOS appended."""
        if not hasattr(self, "eos_label"):
            raise KeyError("EOS label has not been added to label set!")
        return list(x) + [self.eos_label]

    def append_eos_index(self, x):
        """Returns a list version of x, with EOS index appended.
        If the input is a tensor, a tensor is returned."""
        if not hasattr(self, "eos_label"):
            raise KeyError("EOS label has not been added to label set!")
        if paddle.is_tensor(x):
            eos_ind = paddle.Tensor([self.lab2ind[self.eos_label]])
            return paddle.cat([x, eos_ind])
        return list(x) + [self.lab2ind[self.eos_label]]

    def _get_extras(self):
        extras = super()._get_extras()
        if hasattr(self, "bos_label"):
            extras["bos_label"] = self.bos_label
        if hasattr(self, "eos_label"):
            extras["eos_label"] = self.eos_label
        return extras

    def _set_extras(self, extras):
        super()._set_extras(extras)
        if "bos_label" in extras:
            self.bos_label = extras["bos_label"]
        if "eos_label" in extras:
            self.eos_label = extras["eos_label"]


class CTCTextEncoder(TextEncoder):
    """Subclass of TextEncoder which also provides methods to handle CTC blank token.

    add_blank and insert_blank can be used to add <blank> special token to the encoder
    state.

    >>> from speechbrain.dataio.encoder import CTCTextEncoder
    >>> chars = ["a", "b", "c", "d"]
    >>> encoder = CTCTextEncoder()
    >>> encoder.update_from_iterable(chars)
    >>> encoder.add_blank()
    >>> encoder.encode_sequence(chars)
    [0, 1, 2, 3]
    >>> encoder.get_blank_index()
    4
    >>> encoder.decode_ndim([0, 1, 2, 3, 4])
    ['a', 'b', 'c', 'd', '<blank>']

    collapse_labels and collapse_indices_ndim can be used to apply CTC collapsing
    rules:
    >>> encoder.collapse_labels(["a", "a", "b", "c", "d"])
    ['a', 'b', 'c', 'd']
    >>> encoder.collapse_indices_ndim([4, 4, 0, 1, 2, 3, 4, 4]) # 4 is <blank>
    [0, 1, 2, 3]
    """

    def handle_special_labels(self, special_labels):
        # super().handle_special_labels(special_labels)
        # NOTE: blank_label is not necessarily set at all!
        # This is because None is a suitable value.
        # So the test is: hasattr(self, "blank_label")
        # rather than self.blank_label is not None
        # Same thing with unk, see base class.
        if "blank_label" in special_labels:
            self.insert_blank(index=special_labels["blank_label"])

        super().handle_special_labels(special_labels)

    def add_blank(self, blank_label=DEFAULT_BLANK):
        """Add blank symbol to labelset."""
        self.add_label(blank_label)
        self.blank_label = blank_label

    def insert_blank(self, blank_label=DEFAULT_BLANK, index=0):
        """Insert blank symbol at a given labelset."""
        self.insert_label(blank_label, index)
        self.blank_label = blank_label

    def get_blank_index(self):
        """Returns the index to which blank encodes."""
        if not hasattr(self, "blank_label"):
            raise RuntimeError("Blank label is not set!")
        return self.encode_label(self.blank_label)

    def collapse_labels(self, x, merge_repeats=True):
        """Applies the CTC collapsing rules on one label sequence.

        Arguments
        ---------
        x : iterable
            Label sequence on which to operate.
        merge_repeats : bool
            Whether to merge repeated labels before removing blanks.
            In the basic CTC label topology, repeated labels are merged.
            However, in RNN-T, they are not.

        Returns
        -------
        list
            List of labels with collapsing rules applied.
        """
        # This cannot work on arbitrary "ndim", because strings can be
        # infinitely iterated. Iterating "a" produces "a" over and over again.
        if not hasattr(self, "blank_label"):
            raise KeyError("Blank label has not been added")
        if merge_repeats:
            return [
                label
                for i, label in enumerate(x)
                if (i == 0 or label != x[i - 1]) and label != self.blank_label
            ]
        else:
            return [label for label in x if label != self.blank_label]

    def collapse_indices_ndim(self, x, merge_repeats=True):
        """Applies the CTC collapsing rules on arbitrarily label sequence.

        Arguments
        ---------
        x : iterable
            Label sequence on which to operate.
        merge_repeats : bool
            Whether to merge repeated labels before removing blanks.
            In the basic CTC label topology, repeated labels are merged.
            However, in RNN-T, they are not.

        Returns
        -------
        list
            List of labels with collapsing rules applied.
        """
        if not hasattr(self, "blank_label"):
            raise KeyError("Blank label has not been added")
        # Recursively operates on the different dimensions.
        collapsed = []
        for subtensor in x:
            try:
                collapsed.append(
                    self.collapse_indices_ndim(subtensor, merge_repeats)
                )
            except TypeError:  # Not an iterable at next level!
                # So we should rather operate on this dimension.
                break
        else:  # For-else: only enter else if NO break.
            return collapsed
        # We get here if we DID break:
        blank_index = self.lab2ind[self.blank_label]
        if merge_repeats:
            return [
                index
                for i, index in enumerate(x)
                if (i == 0 or index != x[i - 1]) and index != blank_index
            ]
        else:
            return [index for index in x if index != blank_index]

    def _get_extras(self):
        extras = super()._get_extras()
        if hasattr(self, "blank_label"):
            extras["blank_label"] = self.blank_label
        return extras

    def _set_extras(self, extras):
        super()._set_extras(extras)
        if "blank_label" in extras:
            self.blank_label = extras["blank_label"]
