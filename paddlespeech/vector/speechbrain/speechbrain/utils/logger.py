"""Managing the logger, utilities

Author
 * Fang-Pen Lin 2012 https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/
 * Peter Plantinga 2020
 * Aku Rouhe 2020
"""

import sys
import os
import yaml
import tqdm
import logging
import logging.config
import math
import paddle
from speechbrain.utils.data_utils import recursive_update
from speechbrain.utils.superpowers import run_shell

ORDERS_ABBREV = {
    -24: "y",
    -21: "z",
    -18: "a",
    -15: "f",
    -12: "p",
    -9: "n",
    -6: "µ",
    -3: "m",
    0: "",
    3: "k",
    6: "M",
    9: "G",
    12: "T",
    15: "P",
    18: "E",
    21: "Z",
    24: "Y",
}

# Short scale
# Negative powers of ten in lowercase, positive in uppercase
ORDERS_WORDS = {
    -24: "septillionths",
    -21: "sextillionths",
    -18: "quintillionths",
    -15: "quadrillionths",
    -12: "trillionths",
    -9: "billionths",
    -6: "millionths",
    -3: "thousandths",
    0: "",
    3: "Thousand",
    6: "Million",
    9: "Billion",
    12: "Trillion",
    15: "Quadrillion",
    18: "Quintillion",
    21: "Sextillion",
    24: "Septillion",
}


class TqdmCompatibleStreamHandler(logging.StreamHandler):
    """TQDM compatible StreamHandler.

    Writes and prints should be passed through tqdm.tqdm.write
    so that the tqdm progressbar doesn't get messed up.
    """

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            tqdm.tqdm.write(msg, end=self.terminator, file=stream)
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


def setup_logging(
    config_path="log-config.yaml", overrides={}, default_level=logging.INFO,
):
    """Setup logging configuration.

    Arguments
    ---------
    config_path : str
        The path to a logging config file.
    default_level : int
        The level to use if the config file is not found.
    overrides : dict
        A dictionary of the same structure as the config dict
        with any updated values that need to be applied.
    """
    if os.path.exists(config_path):
        with open(config_path, "rt") as f:
            config = yaml.safe_load(f)
        recursive_update(config, overrides)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def format_order_of_magnitude(number, abbreviate=True):
    """Formats number to the appropriate order of magnitude for printing.

    Arguments
    ---------
    number : int, float
        The number to format.
    abbreviate : bool
        Whether to use abbreviations (k,M,G) or words (Thousand, Million,
        Billion). Numbers will be either like: "123.5k" or "123.5 Thousand".

    Returns
    -------
    str
        The formatted number. Note that the order of magnitude token is part
        of the string.

    Example
    -------
    >>> print(format_order_of_magnitude(123456))
    123.5k
    >>> print(format_order_of_magnitude(0.00000123, abbreviate=False))
    1.2 millionths
    >>> print(format_order_of_magnitude(5, abbreviate=False))
    5
    """
    style = ORDERS_ABBREV if abbreviate else ORDERS_WORDS
    precision = "{num:3.1f}"
    order = 3 * math.floor(math.log(math.fabs(number), 1000))
    # Fallback for very large numbers:
    while order not in style and order != 0:
        order = order - math.copysign(3, order)  # Bring 3 units towards 0
    order_token = style[order]
    if order != 0:
        formatted_number = precision.format(num=number / 10 ** order)
    else:
        if isinstance(number, int):
            formatted_number = str(number)
        else:
            # print("number: {}".format(number))
            formatted_number = precision.format(num=number)
    if abbreviate or not order_token:
        return formatted_number + order_token
    else:
        return formatted_number + " " + order_token


def get_environment_description():
    """Returns a string describing the current Python / SpeechBrain environment.

    Useful for making experiments as replicable as possible.

    Returns
    -------
    str
        The string is formatted ready to be written to a file.

    Example
    -------
    >>> get_environment_description().splitlines()[0]
    'SpeechBrain system description'
    """
    python_version_str = "Python version:\n" + sys.version + "\n"
    try:
        freezed, _, _ = run_shell("pip freeze")
        python_packages_str = "Installed Python packages:\n"
        python_packages_str += freezed.decode(errors="replace")
    except OSError:
        python_packages_str = "Could not list python packages with pip freeze"
    try:
        git_hash, _, _ = run_shell("git rev-parse --short HEAD")
        git_str = "Git revision:\n" + git_hash.decode(errors="replace")
    except OSError:
        git_str = "Could not get git revision"
    if paddle.device.is_compiled_with_cuda():
        cuda_str = "Cuda version:\n" + paddle.version.cuda()
    else:
        cuda_str = "CUDA not available"
    result = "SpeechBrain system description\n"
    result += "==============================\n"
    result += python_version_str
    result += "==============================\n"
    result += python_packages_str
    result += "==============================\n"
    result += git_str
    result += "==============================\n"
    result += cuda_str
    return result
