"""SpeechBrain Extended CSV Compatibility."""
from speechbrain.dataio.dataset import DynamicItemDataset
import collections
import csv
import pickle
import logging
import paddle
import re
from paddleaudio.backends.audio import save_wav, sound_file_load, depth_convert
logger = logging.getLogger(__name__)


TORCHAUDIO_FORMATS = ["wav", "flac", "aac", "ogg", "flac", "mp3"]
ITEM_POSTFIX = "_data"

CSVItem = collections.namedtuple("CSVItem", ["data", "format", "opts"])
CSVItem.__doc__ = """The Legacy Extended CSV Data item triplet"""


class ExtendedCSVDataset(DynamicItemDataset):
    """Extended CSV compatibility for DynamicItemDataset.

    Uses the SpeechBrain Extended CSV data format, where the CSV must have an
    'ID' and 'duration' fields.

    The rest of the fields come in triplets:
    ``<name>, <name>_format, <name>_opts``

    These add a <name>_sb_data item in the dict. Additionally, a basic
    DynamicItem (see DynamicItemDataset) is created, which loads the _sb_data
    item.

    Bash-like string replacements with $to_replace are supported.

    NOTE
    ----
    Mapping from legacy interface:

    - csv_file -> csvpath
    - sentence_sorting -> sorting, and "random" is not supported, use e.g.
      ``make_dataloader(..., shuffle = (sorting=="random"))``
    - avoid_if_shorter_than -> min_duration
    - avoid_if_longer_than -> max_duration
    - csv_read -> output_keys, and if you want IDs add "id" as key

    Arguments
    ---------
    csvpath : str, path
        Path to extended CSV.
    replacements : dict
        Used for Bash-like $-prefixed substitution,
        e.g. ``{"data_folder": "/home/speechbrain/data"}``, which would
        transform `$data_folder/utt1.wav` into `/home/speechbain/data/utt1.wav`
    sorting : {"original", "ascending", "descending"}
        Keep CSV order, or sort ascending or descending by duration.
    min_duration : float, int
        Minimum duration in seconds. Discards other entries.
    max_duration : float, int
        Maximum duration in seconds. Discards other entries.
    dynamic_items : list
        Configuration for extra dynamic items produced when fetching an
        example. List of DynamicItems or dicts with keys::
            func: <callable> # To be called
            takes: <list> # key or list of keys of args this takes
            provides: key # key or list of keys that this provides
        NOTE: A dynamic item is automatically added for each CSV data-triplet
    output_keys : list, None
        The list of output keys to produce. You can refer to the names of the
        CSV data-triplets. E.G. if the CSV has: wav,wav_format,wav_opts,
        then the Dataset has a dynamic item output available with key ``"wav"``
        NOTE: If None, read all existing.
    """

    def __init__(
        self,
        csvpath,
        replacements={},
        sorting="original",
        min_duration=0,
        max_duration=36000,
        dynamic_items=[],
        output_keys=[],
    ):
        if sorting not in ["original", "ascending", "descending"]:
            clsname = self.__class__.__name__
            raise ValueError(f"{clsname} doesn't support {sorting} sorting")
        # Load the CSV, init class
        data, di_to_add, data_names = load_sb_extended_csv(
            csvpath, replacements
        )
        super().__init__(data, dynamic_items, output_keys)
        self.pipeline.add_dynamic_items(di_to_add)
        # Handle filtering, sorting:
        reverse = False
        sort_key = None
        if sorting == "ascending" or "descending":
            sort_key = "duration"
        if sorting == "descending":
            reverse = True
        filtered_sorted_ids = self._filtered_sorted_ids(
            key_min_value={"duration": min_duration},
            key_max_value={"duration": max_duration},
            sort_key=sort_key,
            reverse=reverse,
        )
        self.data_ids = filtered_sorted_ids
        # Handle None output_keys (differently than Base)
        if not output_keys:
            self.set_output_keys(data_names)


def load_sb_extended_csv(csv_path, replacements={}):
    """Loads SB Extended CSV and formats string values.

    Uses the SpeechBrain Extended CSV data format, where the
    CSV must have an 'ID' and 'duration' fields.

    The rest of the fields come in triplets:
    ``<name>, <name>_format, <name>_opts``.

    These add a <name>_sb_data item in the dict. Additionally, a
    basic DynamicItem (see DynamicItemDataset) is created, which
    loads the _sb_data item.

    Bash-like string replacements with $to_replace are supported.

    This format has its restriction, but they allow some tasks to
    have loading specified by the CSV.

    Arguments
    ----------
    csv_path : str
        Path to the CSV file.
    replacements : dict
        Optional dict:
        e.g. ``{"data_folder": "/home/speechbrain/data"}``
        This is used to recursively format all string values in the data.

    Returns
    -------
    dict
        CSV data with replacements applied.
    list
        List of DynamicItems to add in DynamicItemDataset.

    """
    with open(csv_path, newline="") as csvfile:
        result = {}
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        variable_finder = re.compile(r"\$([\w.]+)")
        if not reader.fieldnames[0] == "ID":
            raise KeyError(
                "CSV has to have an 'ID' field, with unique ids"
                " for all data points"
            )
        if not reader.fieldnames[1] == "duration":
            raise KeyError(
                "CSV has to have an 'duration' field, "
                "with the length of the data point in seconds."
            )
        if not len(reader.fieldnames[2:]) % 3 == 0:
            raise ValueError(
                "All named fields must have 3 entries: "
                "<name>, <name>_format, <name>_opts"
            )
        names = reader.fieldnames[2::3]
        for row in reader:
            # Make a triplet for each name
            data_point = {}
            # ID:
            data_id = row["ID"]
            del row["ID"]  # This is used as a key in result, instead.
            # Duration:
            data_point["duration"] = float(row["duration"])
            del row["duration"]  # This is handled specially.
            if data_id in result:
                raise ValueError(f"Duplicate id: {data_id}")
            # Replacements:
            # Only need to run these in the actual data,
            # not in _opts, _format
            for key, value in list(row.items())[::3]:
                try:
                    row[key] = variable_finder.sub(
                        lambda match: replacements[match[1]], value
                    )
                except KeyError:
                    raise KeyError(
                        f"The item {value} requires replacements "
                        "which were not supplied."
                    )
            for i, name in enumerate(names):
                triplet = CSVItem(*list(row.values())[i * 3 : i * 3 + 3])
                data_point[name + ITEM_POSTFIX] = triplet
            result[data_id] = data_point
        # Make a DynamicItem for each CSV entry
        # _read_csv_item delegates reading to further
        dynamic_items_to_add = []
        for name in names:
            di = {
                "func": _read_csv_item,
                "takes": name + ITEM_POSTFIX,
                "provides": name,
            }
            dynamic_items_to_add.append(di)
        return result, dynamic_items_to_add, names


def _read_csv_item(item):
    """Reads the different formats supported in SB Extended CSV.

    Delegates to the relevant functions.
    """
    opts = _parse_csv_item_opts(item.opts)
    if item.format in TORCHAUDIO_FORMATS:
        # print("item.data: {}".format(item.data))
        audio, _ = sound_file_load(item.data)
        audio = depth_convert(audio, "float32")
        return audio
        # audio, _ = torchaudio.load(item.data)
        # return audio.squeeze(0)
    elif item.format == "pkl":
        return read_pkl(item.data, opts)
    elif item.format == "string":
        # Just implement string reading here.
        # NOTE: No longer supporting
        # lab2ind mapping like before.
        # Try decoding string
        string = item.data
        try:
            string = string.decode("utf-8")
        except AttributeError:
            pass
        # Splitting elements with ' '
        string = string.split(" ")
        return string
    else:
        raise TypeError(f"Don't know how to read {item.format}")


def _parse_csv_item_opts(entry):
    """Parse the _opts field in a SB Extended CSV item."""
    # Accepting even slightly weirdly formatted entries:
    entry = entry.strip()
    if len(entry) == 0:
        return {}
    opts = {}
    for opt in entry.split(" "):
        opt_name, opt_val = opt.split(":")
        opts[opt_name] = opt_val
    return opts


def read_pkl(file, data_options={}, lab2ind=None):
    """This function reads tensors store in pkl format.

    Arguments
    ---------
    file : str
        The path to file to read.
    data_options : dict, optional
        A dictionary containing options for the reader.
    lab2ind : dict, optional
        Mapping from label to integer indices.

    Returns
    -------
    numpy.array
        The array containing the read signal.
    """

    # Trying to read data
    try:
        with open(file, "rb") as f:
            pkl_element = pickle.load(f)
    except pickle.UnpicklingError:
        err_msg = "cannot read the pkl file %s" % (file)
        raise ValueError(err_msg)

    type_ok = False

    if isinstance(pkl_element, list):

        if isinstance(pkl_element[0], float):
            tensor = paddle.FloatTensor(pkl_element)
            type_ok = True

        if isinstance(pkl_element[0], int):
            tensor = paddle.LongTensor(pkl_element)
            type_ok = True

        if isinstance(pkl_element[0], str):

            # convert string to integer as specified in self.label_dict
            if lab2ind is not None:
                for index, val in enumerate(pkl_element):
                    pkl_element[index] = lab2ind[val]

            tensor = paddle.LongTensor(pkl_element)
            type_ok = True

        if not (type_ok):
            err_msg = (
                "The pkl file %s can only contain list of integers, "
                "floats, or strings. Got %s"
            ) % (file, type(pkl_element[0]))
            raise ValueError(err_msg)
    else:
        tensor = pkl_element

    tensor_type = tensor.dtype

    # Conversion to 32 bit (if needed)
    if tensor_type == "float64":
        tensor = tensor.astype("float32")

    if tensor_type == "int64":
        tensor = tensor.astype("int64")

    return tensor
