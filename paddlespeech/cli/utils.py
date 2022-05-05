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
import inspect
import json
import os
import tarfile
import threading
import time
import uuid
import zipfile
from typing import Any
from typing import Dict

import paddle
import paddleaudio
import requests
import yaml
from paddle.framework import load

from . import download
from .entry import commands
try:
    from .. import __version__
except ImportError:
    __version__ = "0.0.0"  # for develop branch

requests.adapters.DEFAULT_RETRIES = 3

__all__ = [
    'cli_register',
    'get_command',
    'download_and_decompress',
    'load_state_dict_from_url',
    'stats_wrapper',
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
        StatsWorker(
            task='download',
            version=__version__,
            extra_info={
                'download_url': archive['url'],
                'paddle_version': paddle.__version__
            }).start()
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
CONF_HOME = _get_sub_home('conf')


def _md5(text: str):
    '''Calculate the md5 value of the input text.'''
    md5code = hashlib.md5(text.encode())
    return md5code.hexdigest()


class ConfigCache:
    def __init__(self):
        self._data = {}
        self._initialize()
        self.file = os.path.join(CONF_HOME, 'cache.yaml')
        if not os.path.exists(self.file):
            self.flush()
            return

        with open(self.file, 'r') as file:
            try:
                cfg = yaml.load(file, Loader=yaml.FullLoader)
                self._data.update(cfg)
            except Exception as e:
                self.flush()

    @property
    def cache_info(self):
        return self._data['cache_info']

    def _initialize(self):
        # Set default configuration values.
        cache_info = _md5(str(uuid.uuid1())[-12:]) + "-" + str(int(time.time()))
        self._data['cache_info'] = cache_info

    def flush(self):
        '''Flush the current configuration into the configuration file.'''
        with open(self.file, 'w') as file:
            cfg = json.loads(json.dumps(self._data))
            yaml.dump(cfg, file)


stats_api = "http://paddlepaddle.org.cn/paddlehub/stat"
cache_info = ConfigCache().cache_info


class StatsWorker(threading.Thread):
    def __init__(self,
                 task="asr",
                 model=None,
                 version=__version__,
                 extra_info={}):
        threading.Thread.__init__(self)
        self._task = task
        self._model = model
        self._version = version
        self._extra_info = extra_info

    def run(self):
        params = {
            'task': self._task,
            'version': self._version,
            'from': 'ppspeech'
        }
        if self._model:
            params['model'] = self._model

        self._extra_info.update({
            'cache_info': cache_info,
        })
        params.update({"extra": json.dumps(self._extra_info)})

        try:
            requests.get(stats_api, params)
        except Exception:
            pass

        return


def _note_one_stat(cls_name, params={}):
    task = cls_name.replace('Executor', '').lower()  # XXExecutor
    extra_info = {
        'paddle_version': paddle.__version__,
    }

    if 'model' in params:
        model = params['model']
    else:
        model = None

    if 'audio_file' in params:
        try:
            _, sr = paddleaudio.load(params['audio_file'])
        except Exception:
            sr = -1

    if task == 'asr':
        extra_info.update({
            'lang': params['lang'],
            'inp_sr': sr,
            'model_sr': params['sample_rate'],
        })
    elif task == 'st':
        extra_info.update({
            'lang':
            params['src_lang'] + '-' + params['tgt_lang'],
            'inp_sr':
            sr,
            'model_sr':
            params['sample_rate'],
        })
    elif task == 'tts':
        model = params['am']
        extra_info.update({
            'lang': params['lang'],
            'vocoder': params['voc'],
        })
    elif task == 'cls':
        extra_info.update({
            'inp_sr': sr,
        })
    elif task == 'text':
        extra_info.update({
            'sub_task': params['task'],
            'lang': params['lang'],
        })
    else:
        return

    StatsWorker(
        task=task,
        model=model,
        version=__version__,
        extra_info=extra_info, ).start()


def _parse_args(func, *args, **kwargs):
    # FullArgSpec(args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations)
    argspec = inspect.getfullargspec(func)

    keys = argspec[0]
    if keys[0] == 'self':  # Remove self pointer.
        keys = keys[1:]

    default_values = argspec[3]
    values = [None] * (len(keys) - len(default_values))
    values.extend(list(default_values))
    params = dict(zip(keys, values))

    for idx, v in enumerate(args):
        params[keys[idx]] = v
    for k, v in kwargs.items():
        params[k] = v

    return params


def stats_wrapper(executor_func):
    def _warpper(self, *args, **kwargs):
        try:
            _note_one_stat(
                type(self).__name__, _parse_args(executor_func, *args,
                                                 **kwargs))
        except Exception:
            pass
        return executor_func(self, *args, **kwargs)

    return _warpper
