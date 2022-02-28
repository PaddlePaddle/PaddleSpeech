"""Batch collation

Authors
  * Aku Rouhe 2020
"""
import collections
import paddle
import re
import numpy as np
from speechbrain.utils.data_utils import mod_default_collate
from speechbrain.utils.data_utils import recursive_to
from speechbrain.utils.data_utils import batch_pad_right
import collections
np_str_obj_array_pattern = re.compile(r'[SaUO]')
string_classes = (str, bytes)
# from torch.utils.data._utils.collate import default_convert
# from torch.utils.data._utils.pin_memory import (
#     pin_memory as recursive_pin_memory,
# )


PaddedData = collections.namedtuple("PaddedData", ["data", "lengths"])

def default_convert(data):
    r"""
        Function that converts each NumPy array element into a :class:`paddle.Tensor`. If the input is a `Sequence`,
        `Collection`, or `Mapping`, it tries to convert each element inside to a :class:`paddle.Tensor`.
        If the input is not an NumPy array, it is left unchanged.
        This is used as the default function for collation when both `batch_sampler` and
        `batch_size` are NOT defined in :class:`~torch.utils.data.DataLoader`.
        The general input type to output type mapping is similar to that
        of :func:`~torch.utils.data.default_collate`. See the description there for more details.
        Args:
            data: a single data point to be converted
        Examples:
            >>> # Example with `int`
            >>> default_convert(0)
            0
            >>> # Example with NumPy array
            >>> default_convert(np.array([0, 1]))
            tensor([0, 1])
            >>> # Example with NamedTuple
            >>> Point = namedtuple('Point', ['x', 'y'])
            >>> default_convert(Point(0, 0))
            Point(x=0, y=0)
            >>> default_convert(Point(np.array(0), np.array(0)))
            Point(x=tensor(0), y=tensor(0))
            >>> # Example with List
            >>> default_convert([np.array([0, 1]), np.array([2, 3])])
            [tensor([0, 1]), tensor([2, 3])]
    """
    elem_type = type(data)
    if isinstance(data, paddle.Tensor):
        return data
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' \
                and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        # 只支持动态图，在什么时候进入到了静态图？？
        # print("data type: {}".format(type(data)))
        # todo: 进入到了gpu的模式，这里需要调试一下
        paddle.disable_static()
        return paddle.to_tensor(data)
    elif isinstance(data, collections.abc.Mapping):
        try:
            return elem_type({key: default_convert(data[key]) for key in data})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, tuple):
        return [default_convert(d) for d in data]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        try:
            return elem_type([default_convert(d) for d in data])
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return [default_convert(d) for d in data]
    else:
        return data


def recursive_pin_memory(data):
    if isinstance(data, paddle.Tensor):
        return data.pin_memory()
    elif isinstance(data, string_classes):
        return data
    elif isinstance(data, collections.abc.Mapping):
        try:
            return type(data)({k: recursive_pin_memory(sample) for k, sample in data.items()})  # type: ignore[call-arg]
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {k: recursive_pin_memory(sample) for k, sample in data.items()}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return type(data)(*(recursive_pin_memory(sample) for sample in data))
    elif isinstance(data, tuple):
        return [recursive_pin_memory(sample) for sample in data]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence):
        try:
            return type(data)([recursive_pin_memory(sample) for sample in data])  # type: ignore[call-arg]
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return [recursive_pin_memory(sample) for sample in data]
    elif hasattr(data, "pin_memory"):
        return data.pin_memory()
    else:
        return data

class PaddedBatch:
    """Collate_fn when examples are dicts and have variable-length sequences.

    Different elements in the examples get matched by key.
    All numpy tensors get converted to Torch (PyTorch default_convert)
    Then, by default, all paddle.Tensor valued elements get padded and support
    collective pin_memory() and to() calls.
    Regular Python data types are just collected in a list.

    Arguments
    ---------
    examples : list
        List of example dicts, as produced by Dataloader.
    padded_keys : list, None
        (Optional) List of keys to pad on. If None, pad all torch.Tensors
    device_prep_keys : list, None
        (Optional) Only these keys participate in collective memory pinning and moving with
        to().
        If None, defaults to all items with paddle.Tensor values.
    padding_func : callable, optional
        Called with a list of tensors to be padded together. Needs to return
        two tensors: the padded data, and another tensor for the data lengths.
    padding_kwargs : dict
        (Optional) Extra kwargs to pass to padding_func. E.G. mode, value
    apply_default_convert : bool
        Whether to apply PyTorch default_convert (numpy to torch recursively,
        etc.) on all data. Default:True, usually does the right thing.
    nonpadded_stack : bool
        Whether to apply PyTorch-default_collate-like stacking on values that
        didn't get padded. This stacks if it can, but doesn't error out if it
        cannot. Default:True, usually does the right thing.

    Example
    -------
    >>> batch = PaddedBatch([
    ...     {"id": "ex1", "foo": paddle.Tensor([1.])},
    ...     {"id": "ex2", "foo": paddle.Tensor([2., 1.])}])
    >>> # Attribute or key-based access:
    >>> batch.id
    ['ex1', 'ex2']
    >>> batch["id"]
    ['ex1', 'ex2']
    >>> # torch.Tensors get padded
    >>> type(batch.foo)
    <class 'speechbrain.dataio.batch.PaddedData'>
    >>> batch.foo.data
    tensor([[1., 0.],
            [2., 1.]])
    >>> batch.foo.lengths
    tensor([0.5000, 1.0000])
    >>> # Batch supports collective operations:
    >>> _ = batch.to(dtype=torch.half)
    >>> batch.foo.data
    tensor([[1., 0.],
            [2., 1.]], dtype=torch.float16)
    >>> batch.foo.lengths
    tensor([0.5000, 1.0000], dtype=torch.float16)
    >>> # Numpy tensors get converted to torch and padded as well:
    >>> import numpy as np
    >>> batch = PaddedBatch([
    ...     {"wav": np.asarray([1,2,3,4])},
    ...     {"wav": np.asarray([1,2,3])}])
    >>> batch.wav  # +ELLIPSIS
    PaddedData(data=tensor([[1, 2,...
    >>> # Basic stacking collation deals with non padded data:
    >>> batch = PaddedBatch([
    ...     {"spk_id": torch.tensor([1]), "wav": torch.tensor([.1,.0,.3])},
    ...     {"spk_id": torch.tensor([2]), "wav": torch.tensor([.2,.3,-.1])}],
    ...     padded_keys=["wav"])
    >>> batch.spk_id
    tensor([[1],
            [2]])
    >>> # And some data is left alone:
    >>> batch = PaddedBatch([
    ...     {"text": ["Hello"]},
    ...     {"text": ["How", "are", "you?"]}])
    >>> batch.text
    [['Hello'], ['How', 'are', 'you?']]

    """

    def __init__(
        self,
        examples,
        padded_keys=None,
        device_prep_keys=None,
        padding_func=batch_pad_right,
        padding_kwargs={},
        apply_default_convert=True,
        nonpadded_stack=True,
    ):
        self.__length = len(examples)
        # 得到所有的样本关键字
        self.__keys = list(examples[0].keys())
        self.__padded_keys = []
        self.__device_prep_keys = []
        for key in self.__keys:
            # print("process the batch key: {}".format(key))
            values = [example[key] for example in examples]
            # Default convert usually does the right thing (numpy2torch etc.)
            if apply_default_convert:
                values = default_convert(values)
            if (padded_keys is not None and key in padded_keys) or (
                padded_keys is None and isinstance(values[0], paddle.Tensor)
            ):
                # Padding and PaddedData
                self.__padded_keys.append(key)
                # print("start to padded the value")
                padded = PaddedData(*padding_func(values, **padding_kwargs))
                # 每个关键字添加一个属性
                setattr(self, key, padded)
            else:
                # Default PyTorch collate usually does the right thing
                # (convert lists of equal sized tensors to batch tensors, etc.)
                if nonpadded_stack:
                    values = mod_default_collate(values)
                setattr(self, key, values)
            # print("padded keys: {}".format(self.__padded_keys))
            if (device_prep_keys is not None and key in device_prep_keys) or (
                device_prep_keys is None and isinstance(values[0], paddle.Tensor)
            ):
                self.__device_prep_keys.append(key)
    def __len__(self):
        return self.__length

    def __getitem__(self, key):
        if key in self.__keys:
            return getattr(self, key)
        else:
            raise KeyError(f"Batch doesn't have key: {key}")

    def __iter__(self):
        """Iterates over the different elements of the batch.

        Example
        -------
        >>> batch = PaddedBatch([
        ...     {"id": "ex1", "val": paddle.Tensor([1.])},
        ...     {"id": "ex2", "val": paddle.Tensor([2., 1.])}])
        >>> ids, vals = batch
        >>> ids
        ['ex1', 'ex2']
        """
        return iter((getattr(self, key) for key in self.__keys))

    def pin_memory(self):
        """In-place, moves relevant elements to pinned memory."""
        for key in self.__device_prep_keys:
            value = getattr(self, key)
            pinned = recursive_pin_memory(value)
            setattr(self, key, pinned)
        return self

    def to(self, *args, **kwargs):
        """In-place move/cast relevant elements.

        Passes all arguments to paddle.Tensor.to, see its documentation.
        """
        # print("get the to func")
        for key in self.__device_prep_keys:
            value = getattr(self, key)
            moved = recursive_to(value, *args, **kwargs)
            setattr(self, key, moved)
        return self

    def at_position(self, pos):
        """Fetch an item by its position in the batch."""
        key = self.__keys[pos]
        return getattr(self, key)
        
    def keys(self):
        return self.__keys

    @property
    def batchsize(self):
        return self.__length

def padded_batch_collate(examples,
                        padded_keys=None,
                        device_prep_keys=None,
                        padding_func=batch_pad_right,
                        padding_kwargs={},
                        apply_default_convert=True,
                        nonpadded_stack=True,):
    padded_batch_returns = {}
    # 这里第一个参数必须是tensor或者是可以转换为tensor的数据
    # 这里是否已是paddle的一个bug
    # print("batch examples: {}".format(examples))
    padded_batch_returns["placeholder"] = np.array([0])
    padded_batch_returns["batch_value"] = PaddedBatch(examples, 
                            padded_keys=None,
                            device_prep_keys=None,
                            padding_func=batch_pad_right,
                            padding_kwargs={},
                            apply_default_convert=True,
                            nonpadded_stack=True,)
    return padded_batch_returns
    

class BatchsizeGuesser:
    """Try to figure out the batchsize, but never error out

    If this cannot figure out anything else, will fallback to guessing 1

    Example
    -------
    >>> guesser = BatchsizeGuesser()
    >>> # Works with simple tensors:
    >>> guesser(torch.randn((2,3)))
    2
    >>> # Works with sequences of tensors:
    >>> guesser((torch.randn((2,3)), torch.randint(high=5, size=(2,))))
    2
    >>> # Works with PaddedBatch:
    >>> guesser(PaddedBatch([{"wav": [1.,2.,3.]}, {"wav": [4.,5.,6.]}]))
    2
    >>> guesser("Even weird non-batches have a fallback")
    1

    """

    def __init__(self):
        self.method = None

    def __call__(self, batch):
        try:
            return self.method(batch)
        except:  # noqa: E722
            return self.find_suitable_method(batch)

    def find_suitable_method(self, batch):
        """Try the different methods and note which worked"""
        try:
            bs = self.attr_based(batch)
            self.method = self.attr_based
            return bs
        except:  # noqa: E722
            pass
        try:
            bs = self.torch_tensor_bs(batch)
            self.method = self.torch_tensor_bs
            return bs
        except:  # noqa: E722
            pass
        try:
            bs = self.len_of_first(batch)
            self.method = self.len_of_first
            return bs
        except:  # noqa: E722
            pass
        try:
            bs = self.len_of_iter_first(batch)
            self.method = self.len_of_iter_first
            return bs
        except:  # noqa: E722
            pass
        # Last ditch fallback:
        bs = self.fallback(batch)
        self.method = self.fallback(batch)
        return bs

    def attr_based(self, batch):
        return batch.batchsize

    def torch_tensor_bs(self, batch):
        return batch.shape[0]

    def len_of_first(self, batch):
        return len(batch[0])

    def len_of_iter_first(self, batch):
        return len(next(iter(batch)))

    def fallback(self, batch):
        return 1
