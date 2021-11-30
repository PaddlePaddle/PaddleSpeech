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
import hashlib
import os
import sys
import tarfile
import zipfile
from typing import Text

__all__ = [
    "check_md5sum", "getfile_insensitive", "download_multi", "download",
    "unpack", "unzip", "md5file", "print_arguments", "add_arguments",
    "get_commandline_args"
]


def get_commandline_args():
    extra_chars = [
        " ",
        ";",
        "&",
        "(",
        ")",
        "|",
        "^",
        "<",
        ">",
        "?",
        "*",
        "[",
        "]",
        "$",
        "`",
        '"',
        "\\",
        "!",
        "{",
        "}",
    ]

    # Escape the extra characters for shell
    argv = [
        arg.replace("'", "'\\''") if all(char not in arg
                                         for char in extra_chars) else
        "'" + arg.replace("'", "'\\''") + "'" for arg in sys.argv
    ]

    return sys.executable + " " + " ".join(argv)


def print_arguments(args, info=None):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    filename = ""
    if info:
        filename = info["__file__"]
    filename = os.path.basename(filename)
    print(f"----------- {filename} Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("-----------------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)


def md5file(fname):
    hash_md5 = hashlib.md5()
    f = open(fname, "rb")
    for chunk in iter(lambda: f.read(4096), b""):
        hash_md5.update(chunk)
    f.close()
    return hash_md5.hexdigest()


def getfile_insensitive(path):
    """Get the actual file path when given insensitive filename."""
    directory, filename = os.path.split(path)
    directory, filename = (directory or '.'), filename.lower()
    for f in os.listdir(directory):
        newpath = os.path.join(directory, f)
        if os.path.isfile(newpath) and f.lower() == filename:
            return newpath


def download_multi(url, target_dir, extra_args):
    """Download multiple files from url to target_dir."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    print("Downloading %s ..." % url)
    ret_code = os.system("wget -c " + url + ' ' + extra_args + " -P " +
                         target_dir)
    return ret_code


def download(url, md5sum, target_dir):
    """Download file from url to target_dir, and check md5sum."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    filepath = os.path.join(target_dir, url.split("/")[-1])
    if not (os.path.exists(filepath) and md5file(filepath) == md5sum):
        print("Downloading %s ..." % url)
        os.system("wget -c " + url + " -P " + target_dir)
        print("\nMD5 Chesksum %s ..." % filepath)
        if not md5file(filepath) == md5sum:
            raise RuntimeError("MD5 checksum failed.")
    else:
        print("File exists, skip downloading. (%s)" % filepath)
    return filepath


def check_md5sum(filepath: Text, md5sum: Text) -> bool:
    """check md5sum of file.

    Args:
        filepath (Text): [description]
        md5sum (Text): [description]

    Returns:
        bool: same or not.
    """
    return md5file(filepath) == md5sum


def unpack(filepath, target_dir, rm_tar=False):
    """Unpack the file to the target_dir."""
    print("Unpacking %s ..." % filepath)
    tar = tarfile.open(filepath)
    tar.extractall(target_dir)
    tar.close()
    if rm_tar:
        os.remove(filepath)


def unzip(filepath, target_dir, rm_tar=False):
    """Unzip the file to the target_dir."""
    print("Unpacking %s ..." % filepath)
    tar = zipfile.ZipFile(filepath, 'r')
    tar.extractall(target_dir)
    tar.close()
    if rm_tar:
        os.remove(filepath)
