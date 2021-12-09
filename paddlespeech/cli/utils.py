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
import os
import tarfile
import zipfile
from typing import Any
from typing import Dict

from paddle.framework import load

from . import download
from .entry import commands

__all__ = [
    'cli_register',
    'get_command',
    'download_and_decompress',
    'load_state_dict_from_url',
]


def cli_register(name: str, description: str='') -> Any:
    def _warpper(command):
        items = name.split('.')

        com = commands
        for item in items:
            com = com[item]
        com['_entry'] = command
        if description:
            com['_description'] = description
        return command

    return _warpper


def get_command(name: str) -> Any:
    items = name.split('.')
    com = commands
    for item in items:
        com = com[item]

    return com['_entry']


def _get_uncompress_path(filepath: os.PathLike) -> os.PathLike:
    file_dir = os.path.dirname(filepath)
    is_zip_file = False
    if tarfile.is_tarfile(filepath):
        files = tarfile.open(filepath, "r:*")
        file_list = files.getnames()
    elif zipfile.is_zipfile(filepath):
        files = zipfile.ZipFile(filepath, 'r')
        file_list = files.namelist()
        is_zip_file = True
    else:
        return file_dir

    if download._is_a_single_file(file_list):
        rootpath = file_list[0]
        uncompressed_path = os.path.join(file_dir, rootpath)
    elif download._is_a_single_dir(file_list):
        if is_zip_file:
            rootpath = os.path.splitext(file_list[0])[0].split(os.sep)[0]
        else:
            rootpath = os.path.splitext(file_list[0])[0].split(os.sep)[-1]
        uncompressed_path = os.path.join(file_dir, rootpath)
    else:
        rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
        uncompressed_path = os.path.join(file_dir, rootpath)

    files.close()
    return uncompressed_path


def download_and_decompress(archive: Dict[str, str], path: str) -> os.PathLike:
    """
    Download archieves and decompress to specific path.
    """
    if not os.path.isdir(path):
        os.makedirs(path)

    assert 'url' in archive and 'md5' in archive, \
        'Dictionary keys of "url" and "md5" are required in the archive, but got: {}'.format(list(archive.keys()))

    filepath = os.path.join(path, os.path.basename(archive['url']))
    if os.path.isfile(filepath) and download._md5check(filepath,
                                                       archive['md5']):
        uncompress_path = _get_uncompress_path(filepath)
        if not os.path.isdir(uncompress_path):
            download._decompress(filepath)
    else:
        uncompress_path = download.get_path_from_url(archive['url'], path,
                                                     archive['md5'])

    return uncompress_path


def load_state_dict_from_url(url: str, path: str, md5: str=None) -> os.PathLike:
    """
    Download and load a state dict from url
    """
    if not os.path.isdir(path):
        os.makedirs(path)

    download.get_path_from_url(url, path, md5)
    return load(os.path.join(path, os.path.basename(url)))


def _get_user_home():
    return os.path.expanduser('~')


def _get_paddlespcceh_home():
    if 'PPSPEECH_HOME' in os.environ:
        home_path = os.environ['PPSPEECH_HOME']
        if os.path.exists(home_path):
            if os.path.isdir(home_path):
                return home_path
            else:
                raise RuntimeError(
                    'The environment variable PPSPEECH_HOME {} is not a directory.'.
                    format(home_path))
        else:
            return home_path
    return os.path.join(_get_user_home(), '.paddlespeech')


def _get_sub_home(directory):
    home = os.path.join(_get_paddlespcceh_home(), directory)
    if not os.path.exists(home):
        os.makedirs(home)
    return home


PPSPEECH_HOME = _get_paddlespcceh_home()
MODEL_HOME = _get_sub_home('models')
