# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#

# Modified from https://github.com/webdataset/webdataset
# Modified from wenet(https://github.com/wenet-e2e/wenet)
"""A collection of iterators for data transformations.

These functions are plain iterator functions. You can find curried versions
in webdataset.filters, and you can find IterableDataset wrappers in
webdataset.processing.
"""

import io
from fnmatch import fnmatch
import re
import itertools, os, random, sys, time
from functools import reduce, wraps

import numpy as np

from webdataset import autodecode
from . import  utils
from .paddle_utils import PaddleTensor
from .utils import PipelineStage

from .. import backends
from ..compliance import kaldi
import paddle
from ..transform.cmvn import GlobalCMVN
from ..utils.tensor_utils import pad_sequence
from ..transform.spec_augment import time_warp
from ..transform.spec_augment import time_mask
from ..transform.spec_augment import freq_mask

class FilterFunction(object):
    """Helper class for currying pipeline stages.

    We use this roundabout construct becauce it can be pickled.
    """

    def __init__(self, f, *args, **kw):
        """Create a curried function."""
        self.f = f
        self.args = args
        self.kw = kw

    def __call__(self, data):
        """Call the curried function with the given argument."""
        return self.f(data, *self.args, **self.kw)

    def __str__(self):
        """Compute a string representation."""
        return f"<{self.f.__name__} {self.args} {self.kw}>"

    def __repr__(self):
        """Compute a string representation."""
        return f"<{self.f.__name__} {self.args} {self.kw}>"


class RestCurried(object):
    """Helper class for currying pipeline stages.

    We use this roundabout construct because it can be pickled.
    """

    def __init__(self, f):
        """Store the function for future currying."""
        self.f = f

    def __call__(self, *args, **kw):
        """Curry with the given arguments."""
        return FilterFunction(self.f, *args, **kw)


def pipelinefilter(f):
    """Turn the decorated function into one that is partially applied for
    all arguments other than the first."""
    result = RestCurried(f)
    return result


def reraise_exception(exn):
    """Reraises the given exception; used as a handler.

    :param exn: exception
    """
    raise exn


def identity(x):
    """Return the argument."""
    return x


def compose2(f, g):
    """Compose two functions, g(f(x))."""
    return lambda x: g(f(x))


def compose(*args):
    """Compose a sequence of functions (left-to-right)."""
    return reduce(compose2, args)


def pipeline(source, *args):
    """Write an input pipeline; first argument is source, rest are filters."""
    if len(args) == 0:
        return source
    return compose(*args)(source)


def getfirst(a, keys, default=None, missing_is_error=True):
    """Get the first matching key from a dictionary.

    Keys can be specified as a list, or as a string of keys separated by ';'.
    """
    if isinstance(keys, str):
        assert " " not in keys
        keys = keys.split(";")
    for k in keys:
        if k in a:
            return a[k]
    if missing_is_error:
        raise ValueError(f"didn't find {keys} in {list(a.keys())}")
    return default


def parse_field_spec(fields):
    """Parse a specification for a list of fields to be extracted.

    Keys are separated by spaces in the spec. Each key can itself
    be composed of key alternatives separated by ';'.
    """
    if isinstance(fields, str):
        fields = fields.split()
    return [field.split(";") for field in fields]


def transform_with(sample, transformers):
    """Transform a list of values using a list of functions.

    sample: list of values
    transformers: list of functions

    If there are fewer transformers than inputs, or if a transformer
    function is None, then the identity function is used for the
    corresponding sample fields.
    """
    if transformers is None or len(transformers) == 0:
        return sample
    result = list(sample)
    assert len(transformers) <= len(sample)
    for i in range(len(transformers)):  # skipcq: PYL-C0200
        f = transformers[i]
        if f is not None:
            result[i] = f(sample[i])
    return result

###
# Iterators
###

def _info(data, fmt=None, n=3, every=-1, width=50, stream=sys.stderr, name=""):
    """Print information about the samples that are passing through.

    :param data: source iterator
    :param fmt: format statement (using sample dict as keyword)
    :param n: when to stop
    :param every: how often to print
    :param width: maximum width
    :param stream: output stream
    :param name: identifier printed before any output
    """
    for i, sample in enumerate(data):
        if i < n or (every > 0 and (i + 1) % every == 0):
            if fmt is None:
                print("---", name, file=stream)
                for k, v in sample.items():
                    print(k, repr(v)[:width], file=stream)
            else:
                print(fmt.format(**sample), file=stream)
        yield sample


info = pipelinefilter(_info)


def pick(buf, rng):
    k = rng.randint(0, len(buf) - 1)
    sample = buf[k]
    buf[k] = buf[-1]
    buf.pop()
    return sample


def _shuffle(data, bufsize=1000, initial=100, rng=None, handler=None):
    """Shuffle the data in the stream.

    This uses a buffer of size `bufsize`. Shuffling at
    startup is less random; this is traded off against
    yielding samples quickly.

    data: iterator
    bufsize: buffer size for shuffling
    returns: iterator
    rng: either random module or random.Random instance

    """
    if rng is None:
        rng = random.Random(int((os.getpid() + time.time()) * 1e9))
    initial = min(initial, bufsize)
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) < bufsize:
            try:
                buf.append(next(data))  # skipcq: PYL-R1708
            except StopIteration:
                pass
        if len(buf) >= initial:
            yield pick(buf, rng)
    while len(buf) > 0:
        yield pick(buf, rng)


shuffle = pipelinefilter(_shuffle)


class detshuffle(PipelineStage):
    def __init__(self, bufsize=1000, initial=100, seed=0, epoch=-1):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        self.epoch += 1
        rng = random.Random()
        rng.seed((self.seed, self.epoch))
        return _shuffle(src, self.bufsize, self.initial, rng)


def _select(data, predicate):
    """Select samples based on a predicate.

    :param data: source iterator
    :param predicate: predicate (function)
    """
    for sample in data:
        if predicate(sample):
            yield sample


select = pipelinefilter(_select)


def _log_keys(data, logfile=None):
    import fcntl

    if logfile is None or logfile == "":
        for sample in data:
            yield sample
    else:
        with open(logfile, "a") as stream:
            for i, sample in enumerate(data):
                buf = f"{i}\t{sample.get('__worker__')}\t{sample.get('__rank__')}\t{sample.get('__key__')}\n"
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                    stream.write(buf)
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                yield sample


log_keys = pipelinefilter(_log_keys)


def _decode(data, *args, handler=reraise_exception, **kw):
    """Decode data based on the decoding functions given as arguments."""

    decoder = lambda x: autodecode.imagehandler(x) if isinstance(x, str) else x
    handlers = [decoder(x) for x in args]
    f = autodecode.Decoder(handlers, **kw)

    for sample in data:
        assert isinstance(sample, dict), sample
        try:
            decoded = f(sample)
        except Exception as exn:  # skipcq: PYL-W0703
            if handler(exn):
                continue
            else:
                break
        yield decoded


decode = pipelinefilter(_decode)


def _map(data, f, handler=reraise_exception):
    """Map samples."""
    for sample in data:
        try:
            result = f(sample)
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break
        if result is None:
            continue
        if isinstance(sample, dict) and isinstance(result, dict):
            result["__key__"] = sample.get("__key__")
        yield result


map = pipelinefilter(_map)


def _rename(data, handler=reraise_exception, keep=True, **kw):
    """Rename samples based on keyword arguments."""
    for sample in data:
        try:
            if not keep:
                yield {k: getfirst(sample, v, missing_is_error=True) for k, v in kw.items()}
            else:

                def listify(v):
                    return v.split(";") if isinstance(v, str) else v

                to_be_replaced = {x for v in kw.values() for x in listify(v)}
                result = {k: v for k, v in sample.items() if k not in to_be_replaced}
                result.update({k: getfirst(sample, v, missing_is_error=True) for k, v in kw.items()})
                yield result
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break


rename = pipelinefilter(_rename)


def _associate(data, associator, **kw):
    """Associate additional data with samples."""
    for sample in data:
        if callable(associator):
            extra = associator(sample["__key__"])
        else:
            extra = associator.get(sample["__key__"], {})
        sample.update(extra)  # destructive
        yield sample


associate = pipelinefilter(_associate)


def _map_dict(data, handler=reraise_exception, **kw):
    """Map the entries in a dict sample with individual functions."""
    assert len(list(kw.keys())) > 0
    for key, f in kw.items():
        assert callable(f), (key, f)

    for sample in data:
        assert isinstance(sample, dict)
        try:
            for k, f in kw.items():
                sample[k] = f(sample[k])
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break
        yield sample


map_dict = pipelinefilter(_map_dict)


def _to_tuple(data, *args, handler=reraise_exception, missing_is_error=True, none_is_error=None):
    """Convert dict samples to tuples."""
    if none_is_error is None:
        none_is_error = missing_is_error
    if len(args) == 1 and isinstance(args[0], str) and " " in args[0]:
        args = args[0].split()

    for sample in data:
        try:
            result = tuple([getfirst(sample, f, missing_is_error=missing_is_error) for f in args])
            if none_is_error and any(x is None for x in result):
                raise ValueError(f"to_tuple {args} got {sample.keys()}")
            yield result
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break


to_tuple = pipelinefilter(_to_tuple)


def _map_tuple(data, *args, handler=reraise_exception):
    """Map the entries of a tuple with individual functions."""
    args = [f if f is not None else utils.identity for f in args]
    for f in args:
        assert callable(f), f
    for sample in data:
        assert isinstance(sample, (list, tuple))
        sample = list(sample)
        n = min(len(args), len(sample))
        try:
            for i in range(n):
                sample[i] = args[i](sample[i])
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break
        yield tuple(sample)


map_tuple = pipelinefilter(_map_tuple)


def _unlisted(data):
    """Turn batched data back into unbatched data."""
    for batch in data:
        assert isinstance(batch, list), sample
        for sample in batch:
            yield sample


unlisted = pipelinefilter(_unlisted)


def _unbatched(data):
    """Turn batched data back into unbatched data."""
    for sample in data:
        assert isinstance(sample, (tuple, list)), sample
        assert len(sample) > 0
        for i in range(len(sample[0])):
            yield tuple(x[i] for x in sample)


unbatched = pipelinefilter(_unbatched)


def _rsample(data, p=0.5):
    """Randomly subsample a stream of data."""
    assert p >= 0.0 and p <= 1.0
    for sample in data:
        if random.uniform(0.0, 1.0) < p:
            yield sample


rsample = pipelinefilter(_rsample)

slice = pipelinefilter(itertools.islice)


def _extract_keys(source, *patterns, duplicate_is_error=True, ignore_missing=False):
    for sample in source:
        result = []
        for pattern in patterns:
            pattern = pattern.split(";") if isinstance(pattern, str) else pattern
            matches = [x for x in sample.keys() if any(fnmatch("." + x, p) for p in pattern)]
            if len(matches) == 0:
                if ignore_missing:
                    continue
                else:
                    raise ValueError(f"Cannot find {pattern} in sample keys {sample.keys()}.")
            if len(matches) > 1 and duplicate_is_error:
                raise ValueError(f"Multiple sample keys {sample.keys()} match {pattern}.")
            value = sample[matches[0]]
            result.append(value)
        yield tuple(result)


extract_keys = pipelinefilter(_extract_keys)


def _rename_keys(source, *args, keep_unselected=False, must_match=True, duplicate_is_error=True, **kw):
    renamings = [(pattern, output) for output, pattern in args]
    renamings += [(pattern, output) for output, pattern in kw.items()]
    for sample in source:
        new_sample = {}
        matched = {k: False for k, _ in renamings}
        for path, value in sample.items():
            fname = re.sub(r".*/", "", path)
            new_name = None
            for pattern, name in renamings[::-1]:
                if fnmatch(fname.lower(), pattern):
                    matched[pattern] = True
                    new_name = name
                    break
            if new_name is None:
                if keep_unselected:
                    new_sample[path] = value
                continue
            if new_name in new_sample:
                if duplicate_is_error:
                    raise ValueError(f"Duplicate value in sample {sample.keys()} after rename.")
                continue
            new_sample[new_name] = value
        if must_match and not all(matched.values()):
            raise ValueError(f"Not all patterns ({matched}) matched sample keys ({sample.keys()}).")

        yield new_sample


rename_keys = pipelinefilter(_rename_keys)


def decode_bin(stream):
    return stream.read()


def decode_text(stream):
    binary = stream.read()
    return binary.decode("utf-8")


def decode_pickle(stream):
    return pickle.load(stream)


default_decoders = [
    ("*.bin", decode_bin),
    ("*.txt", decode_text),
    ("*.pyd", decode_pickle),
]


def find_decoder(decoders, path):
    fname = re.sub(r".*/", "", path)
    if fname.startswith("__"):
        return lambda x: x
    for pattern, fun in decoders[::-1]:
        if fnmatch(fname.lower(), pattern) or fnmatch("." + fname.lower(), pattern):
            return fun
    return None


def _xdecode(
    source,
    *args,
    must_decode=True,
    defaults=default_decoders,
    **kw,
):
    decoders = list(defaults) + list(args)
    decoders += [("*." + k, v) for k, v in kw.items()]
    for sample in source:
        new_sample = {}
        for path, data in sample.items():
            if path.startswith("__"):
                new_sample[path] = data
                continue
            decoder = find_decoder(decoders, path)
            if decoder is False:
                value = data
            elif decoder is None:
                if must_decode:
                    raise ValueError(f"No decoder found for {path}.")
                value = data
            else:
                if isinstance(data, bytes):
                    data = io.BytesIO(data)
                value = decoder(data)
            new_sample[path] = value
        yield new_sample

xdecode = pipelinefilter(_xdecode)



def _data_filter(source,
           frame_shift=10,
           max_length=10240,
           min_length=10,
           token_max_length=200,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=1):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            source: Iterable[{fname, wav, label, sample_rate}]
            frame_shift: length of frame shift (ms)
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(10ms)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(10ms)

        Returns:
            Iterable[{fname, wav, label, sample_rate}]
    """
    for sample in source:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'label' in sample
        # sample['wav'] is paddle.Tensor, we have 100 frames every second (default)
        num_frames = sample['wav'].shape[1] / sample['sample_rate'] * (1000 / frame_shift)
        if num_frames < min_length:
            continue
        if num_frames > max_length:
            continue
        if len(sample['label']) < token_min_length:
            continue
        if len(sample['label']) > token_max_length:
            continue
        if num_frames != 0:
            if len(sample['label']) / num_frames < min_output_input_ratio:
                continue
            if len(sample['label']) / num_frames > max_output_input_ratio:
                continue
        yield sample

data_filter = pipelinefilter(_data_filter)

def _tokenize(source,
             symbol_table,
             bpe_model=None,
             non_lang_syms=None,
             split_with_space=False):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            source: Iterable[{fname, wav, txt, sample_rate}]

        Returns:
            Iterable[{fname, wav, txt, tokens, label, sample_rate}]
    """
    if non_lang_syms is not None:
        non_lang_syms_pattern = re.compile(r"(\[[^\[\]]+\]|<[^<>]+>|{[^{}]+})")
    else:
        non_lang_syms = {}
        non_lang_syms_pattern = None

    if bpe_model is not None:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)
    else:
        sp = None

    for sample in source:
        assert 'txt' in sample
        txt = sample['txt'].strip()
        if non_lang_syms_pattern is not None:
            parts = non_lang_syms_pattern.split(txt.upper())
            parts = [w for w in parts if len(w.strip()) > 0]
        else:
            parts = [txt]

        label = []
        tokens = []
        for part in parts:
            if part in non_lang_syms:
                tokens.append(part)
            else:
                if bpe_model is not None:
                    tokens.extend(__tokenize_by_bpe_model(sp, part))
                else:
                    if split_with_space:
                        part = part.split(" ")
                    for ch in part:
                        if ch == ' ':
                            ch = "<space>"
                        tokens.append(ch)

        for ch in tokens:
            if ch in symbol_table:
                label.append(symbol_table[ch])
            elif '<unk>' in symbol_table:
                label.append(symbol_table['<unk>'])

        sample['tokens'] = tokens
        sample['label'] = label
        yield sample

tokenize = pipelinefilter(_tokenize)

def _resample(source, resample_rate=16000):
    """ Resample data.
        Inplace operation.

        Args:
            data: Iterable[{fname, wav, label, sample_rate}]
            resample_rate: target resample rate

        Returns:
            Iterable[{fname, wav, label, sample_rate}]
    """
    for sample in source:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        if sample_rate != resample_rate:
            sample['sample_rate'] = resample_rate
            sample['wav'] = paddle.to_tensor(backends.soundfile_backend.resample(
                waveform.numpy(), src_sr = sample_rate, target_sr = resample_rate
            ))
        yield sample

resample = pipelinefilter(_resample)

def _compute_fbank(source,
                  num_mel_bins=80,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0):
    """ Extract fbank

        Args:
            source: Iterable[{fname, wav, label, sample_rate}]
            num_mel_bins: number of mel filter bank
            frame_length: length of one frame (ms)
            frame_shift: length of frame shift (ms)
            dither: value of dither

        Returns:
            Iterable[{fname, feat, label}]
    """
    for sample in source:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'fname' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        waveform = waveform * (1 << 15)
        # Only keep fname, feat, label
        mat = kaldi.fbank(waveform,
                          n_mels=num_mel_bins,
                          frame_length=frame_length,
                          frame_shift=frame_shift,
                          dither=dither,
                          energy_floor=0.0,
                          sr=sample_rate)
        yield dict(fname=sample['fname'], label=sample['label'], feat=mat)


compute_fbank = pipelinefilter(_compute_fbank)

def _spec_aug(source,
            max_w=5, 
            w_inplace=True, 
            w_mode="PIL",
            max_f=30,
            num_f_mask=2, 
            f_inplace=True, 
            f_replace_with_zero=False,
            max_t=40, 
            num_t_mask=2, 
            t_inplace=True, 
            t_replace_with_zero=False,):
    """ Do spec augmentation
        Inplace operation

        Args:
            source: Iterable[{fname, feat, label}]
            max_w: max width of time warp
            w_inplace: whether to inplace the original data while time warping
            w_mode: time warp mode
            max_f: max width of freq mask
            num_f_mask: number of freq mask to apply
            f_inplace: whether to inplace the original data while frequency masking
            f_replace_with_zero: use zero to mask
            max_t: max width of time mask
            num_t_mask: number of time mask to apply
            t_inplace: whether to inplace the original data while time masking
            t_replace_with_zero: use zero to mask
            
        Returns
            Iterable[{fname, feat, label}]
     """
    for sample in source:
        x = sample['feat']
        x = x.numpy()
        x = time_warp(x, max_time_warp=max_w, inplace = w_inplace, mode= w_mode)
        x = freq_mask(x, F = max_f, n_mask = num_f_mask, inplace = f_inplace, replace_with_zero = f_replace_with_zero)
        x = time_mask(x, T = max_t, n_mask = num_t_mask, inplace = t_inplace, replace_with_zero = t_replace_with_zero)
        sample['feat'] = paddle.to_tensor(x, dtype=paddle.float32)
        yield sample

spec_aug = pipelinefilter(_spec_aug)


def _sort(source, sort_size=500):
    """ Sort the data by feature length.
        Sort is used after shuffle and before batch, so we can group
        utts with similar lengths into a batch, and `sort_size` should
        be less than `shuffle_size`

        Args:
            source: Iterable[{fname, feat, label}]
            sort_size: buffer size for sort

        Returns:
            Iterable[{fname, feat, label}]
    """

    buf = []
    for sample in source:
        buf.append(sample)
        if len(buf) >= sort_size:
            buf.sort(key=lambda x: x['feat'].shape[0])
            for x in buf:
                yield x
            buf = []
    # The sample left over
    buf.sort(key=lambda x: x['feat'].shape[0])
    for x in buf:
        yield x

sort = pipelinefilter(_sort)

def _batched(source, batch_size=16):
    """ Static batch the data by `batch_size`

        Args:
            data: Iterable[{fname, feat, label}]
            batch_size: batch size

        Returns:
            Iterable[List[{fname, feat, label}]]
    """
    buf = []
    for sample in source:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf

batched = pipelinefilter(_batched)

def dynamic_batched(source, max_frames_in_batch=12000):
    """ Dynamic batch the data until the total frames in batch
        reach `max_frames_in_batch`

        Args:
            source: Iterable[{fname, feat, label}]
            max_frames_in_batch: max_frames in one batch

        Returns:
            Iterable[List[{fname, feat, label}]]
    """
    buf = []
    longest_frames = 0
    for sample in source:
        assert 'feat' in sample
        assert isinstance(sample['feat'], paddle.Tensor)
        new_sample_frames = sample['feat'].size(0)
        longest_frames = max(longest_frames, new_sample_frames)
        frames_after_padding = longest_frames * (len(buf) + 1)
        if frames_after_padding > max_frames_in_batch:
            yield buf
            buf = [sample]
            longest_frames = new_sample_frames
        else:
            buf.append(sample)
    if len(buf) > 0:
        yield buf


def _padding(source):
    """ Padding the data into training data

        Args:
            source: Iterable[List[{fname, feat, label}]]

        Returns:
            Iterable[Tuple(fname, feats, labels, feats lengths, label lengths)]
    """
    for sample in source:
        assert isinstance(sample, list)
        feats_length = paddle.to_tensor([x['feat'].shape[0] for x in sample],
                                    dtype="int64")
        order = paddle.argsort(feats_length, descending=True)
        feats_lengths = paddle.to_tensor(
            [sample[i]['feat'].shape[0] for i in order], dtype="int64")
        sorted_feats = [sample[i]['feat'] for i in order]
        sorted_keys = [sample[i]['fname'] for i in order]
        sorted_labels = [
            paddle.to_tensor(sample[i]['label'], dtype="int32") for i in order
        ]
        label_lengths = paddle.to_tensor([x.shape[0] for x in sorted_labels],
                                     dtype="int64")
        padded_feats = pad_sequence(sorted_feats,
                                    batch_first=True,
                                    padding_value=0)
        padding_labels = pad_sequence(sorted_labels,
                                      batch_first=True,
                                      padding_value=-1)

        yield (sorted_keys, padded_feats, feats_lengths, padding_labels, 
               label_lengths)

padding = pipelinefilter(_padding)

def _cmvn(source, cmvn_file):
    global_cmvn = GlobalCMVN(cmvn_file)
    for batch in source:
        sorted_keys, padded_feats, feats_lengths, padding_labels, label_lengths = batch
        padded_feats = padded_feats.numpy()
        padded_feats = global_cmvn(padded_feats)
        padded_feats = paddle.to_tensor(padded_feats, dtype=paddle.float32)
        yield (sorted_keys, padded_feats, feats_lengths, padding_labels, 
           label_lengths)

cmvn = pipelinefilter(_cmvn)

def _placeholder(source):
    for data in source:
        yield data

placeholder = pipelinefilter(_placeholder)