# Copyright (c) 2023 speechbrain Authors. All Rights Reserved.
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
# 
# Modified from speechbrain 2023 (https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/dataio/dataio.py)
"""
Data reading and writing.

Authors
 * Mirco Ravanelli 2020
 * Aku Rouhe 2020
 * Ju-Chieh Chou 2020
 * Samuele Cornell 2020
 * Abdel HEBA 2020
"""
import csv
import hashlib
import json
import logging
import os
import pickle
import re
import time

import numpy as np
import soundfile
logger = logging.getLogger(__name__)
import paddle


def load_data_json(json_path, replacements={}):
    """Loads JSON and recursively formats string values.

    Arguments
    ----------
    json_path : str
        Path to CSV file.
    replacements : dict
        (Optional dict), e.g., {"data_folder": "/home/PaddleSpeech/data"}.
        This is used to recursively format all string values in the data.

    Returns
    -------
    dict
        JSON data with replacements applied.


    """
    with open(json_path, "r") as f:
        out_json = json.load(f)
    _recursive_format(out_json, replacements)
    return out_json


def _recursive_format(data, replacements):
    # Data: dict or list, replacements : dict
    # Replaces string keys in replacements by their values
    # at all levels of data (in str values)
    # Works in-place.
    if isinstance(data, dict):
        for key, item in data.items():
            if isinstance(item, dict) or isinstance(item, list):
                _recursive_format(item, replacements)
            elif isinstance(item, str):
                data[key] = item.format_map(replacements)
            # If not dict, list or str, do nothing
    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict) or isinstance(item, list):
                _recursive_format(item, replacements)
            elif isinstance(item, str):
                data[i] = item.format_map(replacements)
            # If not dict, list or str, do nothing


def load_data_csv(csv_path, replacements={}):
    """Loads CSV and formats string values.

    Uses the legacy CSV data format, where the CSV must have an
    'ID' field.
    If there is a field called duration, it is interpreted as a float.
    The rest of the fields are left as they are (legacy _format and _opts fields
    are not used to load the data in any special way).

    Bash-like string replacements with $to_replace are supported.

    Arguments
    ----------
    csv_path : str
        Path to CSV file.
    replacements : dict
        (Optional dict), e.g., {"data_folder": "/home/PaddleSpeech/data"}
        This is used to recursively format all string values in the data.

    Returns
    -------
    dict
        CSV data with replacements applied.
    """

    with open(csv_path, newline="") as csvfile:
        result = {}
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        variable_finder = re.compile(r"\$([\w.]+)")
        for row in reader:
            # ID:
            try:
                data_id = row["ID"]
                del row["ID"]  # This is used as a key in result, instead.
            except KeyError:
                raise KeyError("CSV has to have an 'ID' field, with unique ids"
                               " for all data points")
            if data_id in result:
                raise ValueError(f"Duplicate id: {data_id}")
            # Replacements:
            for key, value in row.items():
                try:
                    row[key] = variable_finder.sub(
                        lambda match: str(replacements[match[1]]), value)
                except KeyError:
                    raise KeyError(f"The item {value} requires replacements "
                                   "which were not supplied.")
            # Duration:
            if "duration" in row:
                row["duration"] = float(row["duration"])
            result[data_id] = row
    return result


def read_audio(waveforms_obj):
    """General audio loading, based on a custom notation.

    Expected use case is in conjunction with Datasets
    specified by JSON.

    The custom notation:

    The annotation can be just a path to a file:
    "/path/to/wav1.wav"

    Or can specify more options in a dict:
    {"file": "/path/to/wav2.wav",
    "start": 8000,
    "stop": 16000
    }

    Arguments
    ----------
    waveforms_obj : str, dict
        Audio reading annotation, see above for format.

    Returns
    -------
    paddle.Tensor
        Audio tensor with shape: (samples, ).
    """
    if isinstance(waveforms_obj, str):
        audio, _ = soundfile.read(waveforms_obj, dtype="float32")
        return audio

    path = waveforms_obj["file"]
    start = waveforms_obj.get("start", 0)
    # Default stop to start -> if not specified, num_frames becomes 0
    stop = waveforms_obj.get("stop", start)
    num_frames = stop - start
    audio, fs = soundfile.read(
        path, start=start, stop=start + num_frames, dtype="float32")
    return audio


def read_audio_multichannel(waveforms_obj):
    """General audio loading, based on a custom notation.

    Expected use case is in conjunction with Datasets
    specified by JSON.

    The custom notation:

    The annotation can be just a path to a file:
    "/path/to/wav1.wav"

    Multiple (possibly multi-channel) files can be specified, as long as they
    have the same length:
    {"files": [
        "/path/to/wav1.wav",
        "/path/to/wav2.wav"
        ]
    }

    Or you can specify a single file more succinctly:
    {"files": "/path/to/wav2.wav"}

    Offset number samples and stop number samples also can be specified to read
    only a segment within the files.
    {"files": [
        "/path/to/wav1.wav",
        "/path/to/wav2.wav"
        ]
    "start": 8000
    "stop": 16000
    }

    Arguments
    ----------
    waveforms_obj : str, dict
        Audio reading annotation, see above for format.

    Returns
    -------
    paddle.Tensor
        Audio tensor with shape: (samples, ).
    """
    if isinstance(waveforms_obj, str):
        audio, _ = soundfile.read(waveforms_obj, dtype="float32")
        audio = paddle.to_tensor(audio)
        return audio

    files = waveforms_obj["files"]
    if not isinstance(files, list):
        files = [files]

    waveforms = []
    start = waveforms_obj.get("start", 0)
    # Default stop to start -> if not specified, num_frames becomes 0
    stop = waveforms_obj.get("stop", start - 1)
    num_frames = stop - start
    for f in files:
        audio, fs = soundfile.read(
            path, start=start, stop=start + num_frames, dtype="float32")
        audio = paddle.to_tensor(audio)
        waveforms.append(audio)

    out = paddle.concat(waveforms, 0)
    return out


def write_audio(filepath, audio, samplerate):
    """Write audio on disk. It is basically a wrapper to support saving
    audio signals in format (audio, channels).

    Arguments
    ---------
    filepath: path
        Path where to save the audio file.
    audio : paddle.Tensor
        Audio file in the expected format (signal, channels).
    samplerate: int
        Sample rate (e.g., 16000).

    """
    if len(audio.shape) == 2:
        audio = audio.transpose([1, 0])
    elif len(audio.shape) == 1:
        audio = audio.unsqueeze(0)

    soundfile.write(filepath, audio, samplerate)


def load_pickle(pickle_path):
    """Utility function for loading .pkl pickle files.

    Arguments
    ---------
    pickle_path : str
        Path to pickle file.

    Returns
    -------
    out : object
        Python object loaded from pickle.
    """
    with open(pickle_path, "rb") as f:
        out = pickle.load(f)
    return out


def to_floatTensor(x: (list, tuple, np.ndarray)):
    """
    Arguments
    ---------
    x : (list, tuple, np.ndarray)
        Input data to be converted to paddle float.

    Returns
    -------
    tensor : paddle.tensor
        Data now in paddle.tensor float datatype.
    """
    return paddle.to_tensor(x, dtype='float32')


def to_doubleTensor(x: (list, tuple, np.ndarray)):
    """
    Arguments
    ---------
    x : (list, tuple, np.ndarray)
        Input data to be converted to paddle double.

    Returns
    -------
    tensor : paddle.tensor
        Data now in paddle.tensor double datatype.
    """
    return paddle.to_tensor(x, dtype='float64')


def to_longTensor(x: (list, tuple, np.ndarray)):
    """
    Arguments
    ---------
    x : (list, tuple, np.ndarray)
        Input data to be converted to paddle long.

    Returns
    -------
    tensor : paddle.tensor
        Data now in paddle.tensor long datatype.
    """
    return paddle.to_tensor(x, dtype='int64')


def convert_index_to_lab(batch, ind2lab):
    """Convert a batch of integer IDs to string labels.

    Arguments
    ---------
    batch : list
        List of lists, a batch of sequences.
    ind2lab : dict
        Mapping from integer IDs to labels.

    Returns
    -------
    list
        List of lists, same size as batch, with labels from ind2lab.

    """
    return [[ind2lab[int(index)] for index in seq] for seq in batch]


def relative_time_to_absolute(batch, relative_lens, rate):
    """Converts relative length to the absolute duration.

    Operates on batch level.

    Arguments
    ---------
    batch : paddle.tensor
        Sequences to determine the duration for.
    relative_lens : paddle.tensor
        The relative length of each sequence in batch. The longest sequence in
        the batch needs to have relative length 1.0.
    rate : float
        The rate at which sequence elements occur in real-world time. Sample
        rate, if batch is raw wavs (recommended) or 1/frame_shift if batch is
        features. This has to have 1/s as the unit.

    Returns
    ------:
    paddle.tensor
        Duration of each sequence in seconds.

    """
    max_len = batch.shape[1]
    durations = paddle.round(relative_lens * max_len) / rate
    return durations


class IterativeCSVWriter:
    """Write CSV files a line at a time.

    Arguments
    ---------
    outstream : file-object
        A writeable stream
    data_fields : list
        List of the optional keys to write. Each key will be expanded, 
        producing three fields: key, key_format, key_opts.
    """

    def __init__(self, outstream, data_fields, defaults={}):
        self._outstream = outstream
        self.fields = ["ID", "duration"] + self._expand_data_fields(data_fields)
        self.defaults = defaults
        self._outstream.write(",".join(self.fields))

    def set_default(self, field, value):
        """Sets a default value for the given CSV field.

        Arguments
        ---------
        field : str
            A field in the CSV.
        value
            The default value.
        """
        if field not in self.fields:
            raise ValueError(f"{field} is not a field in this CSV!")
        self.defaults[field] = value

    def write(self, *args, **kwargs):
        """Writes one data line into the CSV.

        Arguments
        ---------
        *args
            Supply every field with a value in positional form OR.
        **kwargs
            Supply certain fields by key. The ID field is mandatory for all
            lines, but others can be left empty.
        """
        if args and kwargs:
            raise ValueError(
                "Use either positional fields or named fields, but not both.")
        if args:
            if len(args) != len(self.fields):
                raise ValueError("Need consistent fields")
            to_write = [str(arg) for arg in args]
        if kwargs:
            if "ID" not in kwargs:
                raise ValueError("I'll need to see some ID")
            full_vals = self.defaults.copy()
            full_vals.update(kwargs)
            to_write = [str(full_vals.get(field, "")) for field in self.fields]
        self._outstream.write("\n")
        self._outstream.write(",".join(to_write))

    def write_batch(self, *args, **kwargs):
        """Writes a batch of lines into the CSV.

        Here each argument should be a list with the same length.

        Arguments
        ---------
        *args
            Supply every field with a value in positional form OR.
        **kwargs
            Supply certain fields by key. The ID field is mandatory for all
            lines, but others can be left empty.
        """
        if args and kwargs:
            raise ValueError(
                "Use either positional fields or named fields, but not both.")
        if args:
            if len(args) != len(self.fields):
                raise ValueError("Need consistent fields")
            for arg_row in zip(*args):
                self.write(*arg_row)
        if kwargs:
            if "ID" not in kwargs:
                raise ValueError("I'll need to see some ID")
            keys = kwargs.keys()
            for value_row in zip(*kwargs.values()):
                kwarg_row = dict(zip(keys, value_row))
                self.write(**kwarg_row)

    @staticmethod
    def _expand_data_fields(data_fields):
        expanded = []
        for data_field in data_fields:
            expanded.append(data_field)
            expanded.append(data_field + "_format")
            expanded.append(data_field + "_opts")
        return expanded


def write_txt_file(data, filename, sampling_rate=None):
    """Write data in text format.

    Arguments
    ---------
    data : str, list, paddle.tensor, numpy.ndarray
        The data to write in the text file.
    filename : str
        Path to file where to write the data.
    sampling_rate : None
        Not used, just here for interface compatibility.

    Returns
    -------
    None

    """
    del sampling_rate  # Not used.
    # Check if the path of filename exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as fout:
        if isinstance(data, paddle.Tensor):
            data = data.tolist()
        if isinstance(data, np.ndarray):
            data = data.tolist()
        if isinstance(data, list):
            for line in data:
                print(line, file=fout)
        if isinstance(data, str):
            print(data, file=fout)


def write_stdout(data, filename=None, sampling_rate=None):
    """Write data to standard output.

    Arguments
    ---------
    data : str, list, paddle.Tensor, numpy.ndarray
        The data to write in the text file.
    filename : None
        Not used, just here for compatibility.
    sampling_rate : None
        Not used, just here for compatibility.

    Returns
    -------
    None

    """
    # Managing paddle.Tensor
    if isinstance(data, paddle.Tensor):
        data = data.tolist()
    # Managing np.ndarray
    if isinstance(data, np.ndarray):
        data = data.tolist()
    if isinstance(data, list):
        for line in data:
            print(line)
    if isinstance(data, str):
        print(data)


def length_to_mask(length, max_len=None, dtype=None, device=None):
    """Creates a binary mask for each sequence.
    Arguments
    ---------
    length : LongTensor
        Containing the length of each sequence in the batch. Must be 1D.
    max_len : int
        Max length for the mask, also the size of the second dimension.
    dtype : dtype, default: None
        The dtype of the generated mask.
    device: device, default: None
        The device to put the mask variable.

    Returns
    -------
    mask : tensor
        The binary mask.

    """
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().long().item()  # using arange to generate mask
    mask = paddle.arange(
        max_len, dtype=length.dtype).expand(
            [len(length), max_len]) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    if device is None:
        device = length.device

    mask = paddle.to_tensor(mask, dtype=dtype)
    return mask


def read_kaldi_lab(kaldi_ali, kaldi_lab_opts):
    """Read labels in kaldi format.

    Uses kaldi IO.

    Arguments
    ---------
    kaldi_ali : str
        Path to directory where kaldi alignments are stored.
    kaldi_lab_opts : str
        A string that contains the options for reading the kaldi alignments.

    Returns
    -------
    lab : dict
        A dictionary containing the labels.

    Note
    ----
    This depends on kaldi-io-for-python. Install it separately.
    See: https://github.com/vesis84/kaldi-io-for-python
    ```
    """
    # EXTRA TOOLS
    try:
        import kaldi_io
    except ImportError:
        raise ImportError("Could not import kaldi_io. Install it to use this.")
    # Reading the Kaldi labels
    lab = {
        k: v
        for k, v in kaldi_io.read_vec_int_ark(
            "gunzip -c " + kaldi_ali + "/ali*.gz | " + kaldi_lab_opts + " " +
            kaldi_ali + "/final.mdl ark:- ark:-|")
    }
    return lab


def get_md5(file):
    """Get the md5 checksum of an input file.

    Arguments
    ---------
    file : str
        Path to file for which compute the checksum.

    Returns
    -------
    md5
        Checksum for the given filepath.
    """
    # Lets read stuff in 64kb chunks!
    BUF_SIZE = 65536
    md5 = hashlib.md5()
    # Computing md5
    with open(file, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()


def save_md5(files, out_file):
    """Saves the md5 of a list of input files as a pickled dict into a file.

    Arguments
    ---------
    files : list
        List of input files from which we will compute the md5.
    outfile : str
        The path where to store the output pkl file.

    Returns
    -------
    None
    """
    # Initialization of the dictionary
    md5_dict = {}
    # Computing md5 for all the files in the list
    for file in files:
        md5_dict[file] = get_md5(file)
    # Saving dictionary in pkl format
    save_pkl(md5_dict, out_file)


def save_pkl(obj, file):
    """Save an object in pkl format.

    Arguments
    ---------
    obj : object
        Object to save in pkl format
    file : str
        Path to the output file
    sampling_rate : int
        Sampling rate of the audio file, TODO: this is not used?

    """
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(file):
    """Loads a pkl file.

    For an example, see `save_pkl`.

    Arguments
    ---------
    file : str
        Path to the input pkl file.

    Returns
    -------
    The loaded object.
    """

    # Deals with the situation where two processes are trying
    # to access the same label dictionary by creating a lock
    count = 100
    while count > 0:
        if os.path.isfile(file + ".lock"):
            time.sleep(1)
            count -= 1
        else:
            break

    try:
        open(file + ".lock", "w").close()
        with open(file, "rb") as f:
            return pickle.load(f)
    finally:
        if os.path.isfile(file + ".lock"):
            os.remove(file + ".lock")


def prepend_bos_token(label, bos_index):
    """Create labels with <bos> token at the beginning.

    Arguments
    ---------
    label : IntTensor
        Containing the original labels. Must be of size: [batch_size, max_length].
    bos_index : int
        The index for <bos> token.

    Returns
    -------
    new_label : tensor
        The new label with <bos> at the beginning.

    """
    new_label = label.long().clone()
    batch_size = label.shape[0]

    bos = new_label.new_zeros(batch_size, 1).fill_(bos_index)
    new_label = paddle.concat([bos, new_label], axis=1)
    return new_label


def append_eos_token(label, length, eos_index):
    """Create labels with <eos> token appended.

    Arguments
    ---------
    label : IntTensor
        Containing the original labels. Must be of size: [batch_size, max_length]
    length : LongTensor
        Containing the original length of each label sequences. Must be 1D.
    eos_index : int
        The index for <eos> token.

    Returns
    -------
    new_label : tensor
        The new label with <eos> appended.

    """
    new_label = paddle.to_tensor(label, dtype="int32").clone()
    batch_size = label.shape[0]

    pad = paddle.zeros([batch_size, 1], dtype=new_label.dtype)

    new_label = paddle.concat([new_label, pad], dim=1)
    new_label[paddle.arange(batch_size), paddle.to_tensor(
        length, dtype="int64")] = eos_index
    return new_label


def merge_char(sequences, space="_"):
    """Merge characters sequences into word sequences.

    Arguments
    ---------
    sequences : list
        Each item contains a list, and this list contains a character sequence.
    space : string
        The token represents space. Default: _

    Returns
    -------
    The list contains word sequences for each sentence.

    """
    results = []
    for seq in sequences:
        words = "".join(seq).split(space)
        results.append(words)
    return results


def merge_csvs(data_folder, csv_lst, merged_csv):
    """Merging several csv files into one file.

    Arguments
    ---------
    data_folder : string
        The folder to store csv files to be merged and after merging.
    csv_lst : list
        Filenames of csv file to be merged.
    merged_csv : string
        The filename to write the merged csv file.

    """
    write_path = os.path.join(data_folder, merged_csv)
    if os.path.isfile(write_path):
        logger.info("Skipping merging. Completed in previous run.")
    with open(os.path.join(data_folder, csv_lst[0])) as f:
        header = f.readline()
    lines = []
    for csv_file in csv_lst:
        with open(os.path.join(data_folder, csv_file)) as f:
            for i, line in enumerate(f):
                if i == 0:
                    # Checking header
                    if line != header:
                        raise ValueError("Different header for "
                                         f"{csv_lst[0]} and {csv}.")
                    continue
                lines.append(line)
    with open(write_path, "w") as f:
        f.write(header)
        for line in lines:
            f.write(line)
    logger.info(f"{write_path} is created.")


def split_word(sequences, space="_"):
    """Split word sequences into character sequences.

    Arguments
    ---------
    sequences : list
        Each item contains a list, and this list contains a words sequence.
    space : string
        The token represents space. Default: _

    Returns
    -------
    The list contains word sequences for each sentence.

    """
    results = []
    for seq in sequences:
        chars = list(space.join(seq))
        results.append(chars)
    return results
