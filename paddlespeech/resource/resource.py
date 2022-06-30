# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from collections import OrderedDict
from typing import Dict
from typing import List
from typing import Optional

from ..cli.utils import download_and_decompress
from ..utils.dynamic_import import dynamic_import
from ..utils.env import MODEL_HOME
from .model_alias import model_alias

task_supported = ['asr', 'cls', 'st', 'text', 'tts', 'vector', 'kws']
model_format_supported = ['dynamic', 'static', 'onnx']
inference_mode_supported = ['online', 'offline']


class CommonTaskResource:
    def __init__(self, task: str, model_format: str='dynamic', **kwargs):
        assert task in task_supported, 'Arg "task" must be one of {}.'.format(
            task_supported)
        assert model_format in model_format_supported, 'Arg "model_format" must be one of {}.'.format(
            model_format_supported)

        self.task = task
        self.model_format = model_format
        self.pretrained_models = self._get_pretrained_models()

        if 'inference_mode' in kwargs:
            assert kwargs[
                'inference_mode'] in inference_mode_supported, 'Arg "inference_mode" must be one of {}.'.format(
                    inference_mode_supported)
            self._inference_mode_filter(kwargs['inference_mode'])

        # Initialize after model and version had been set.
        self.model_tag = None
        self.version = None
        self.res_dict = None
        self.res_dir = None

        if self.task == 'tts':
            # For vocoder
            self.voc_model_tag = None
            self.voc_version = None
            self.voc_res_dict = None
            self.voc_res_dir = None

    def set_task_model(self,
                       model_tag: str,
                       model_type: int=0,
                       skip_download: bool=False,
                       version: Optional[str]=None):
        """Set model tag and version of current task.

        Args:
            model_tag (str): Model tag.
            model_type (int): 0 for acoustic model otherwise vocoder in tts task.
            version (Optional[str], optional): Version of pretrained model. Defaults to None.
        """
        assert model_tag in self.pretrained_models, \
            "Can't find \"{}\" in resource. Model name must be one of {}".format(model_tag, list(self.pretrained_models.keys()))

        if version is None:
            version = self._get_default_version(model_tag)

        assert version in self.pretrained_models[model_tag], \
            "Can't find version \"{}\" in \"{}\". Model name must be one of {}".format(
                version, model_tag, list(self.pretrained_models[model_tag].keys()))

        if model_type == 0:
            self.model_tag = model_tag
            self.version = version
            self.res_dict = self.pretrained_models[model_tag][version]
            self._format_path(self.res_dict)
            if not skip_download:
                self.res_dir = self._fetch(self.res_dict,
                                           self._get_model_dir(model_type))
        else:
            assert self.task == 'tts', 'Vocoder will only be used in tts task.'
            self.voc_model_tag = model_tag
            self.voc_version = version
            self.voc_res_dict = self.pretrained_models[model_tag][version]
            self._format_path(self.voc_res_dict)
            if not skip_download:
                self.voc_res_dir = self._fetch(self.voc_res_dict,
                                               self._get_model_dir(model_type))

    @staticmethod
    def get_model_class(model_name) -> List[object]:
        """Dynamic import model class.
        Args:
            model_name (str): Model name.

        Returns:
            List[object]: Return a list of model class.
        """
        assert model_name in model_alias, 'No model classes found for "{}"'.format(
            model_name)

        ret = []
        for import_path in model_alias[model_name]:
            ret.append(dynamic_import(import_path))

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def get_versions(self, model_tag: str) -> List[str]:
        """List all available versions.

        Args:
            model_tag (str): Model tag.

        Returns:
            List[str]: Version list of model.
        """
        return list(self.pretrained_models[model_tag].keys())

    def _get_default_version(self, model_tag: str) -> str:
        """Get default version of model.

        Args:
            model_tag (str): Model tag.

        Returns:
            str: Default version.
        """
        return self.get_versions(model_tag)[-1]  # get latest version

    def _get_model_dir(self, model_type: int=0) -> os.PathLike:
        """Get resource directory.

        Args:
            model_type (int): 0 for acoustic model otherwise vocoder in tts task.

        Returns:
            os.PathLike: Directory of model resource.
        """
        if model_type == 0:
            model_tag = self.model_tag
            version = self.version
        else:
            model_tag = self.voc_model_tag
            version = self.voc_version

        return os.path.join(MODEL_HOME, model_tag, version)

    def _get_pretrained_models(self) -> Dict[str, str]:
        """Get all available models for current task.

        Returns:
            Dict[str, str]: A dictionary with model tag and resources info.
        """
        try:
            import_models = '{}_{}_pretrained_models'.format(self.task,
                                                             self.model_format)
            exec('from .pretrained_models import {}'.format(import_models))
            models = OrderedDict(locals()[import_models])
        except Exception as e:
            models = OrderedDict({})  # no models.
        finally:
            return models

    def _inference_mode_filter(self, inference_mode: Optional[str]):
        """Filter models dict based on inference_mode.

        Args:
            inference_mode (Optional[str]): 'online', 'offline' or None.
        """
        if inference_mode is None:
            return

        if self.task == 'asr':
            online_flags = [
                'online' in model_tag
                for model_tag in self.pretrained_models.keys()
            ]
            for online_flag, model_tag in zip(
                    online_flags, list(self.pretrained_models.keys())):
                if inference_mode == 'online' and online_flag:
                    continue
                elif inference_mode == 'offline' and not online_flag:
                    continue
                else:
                    del self.pretrained_models[model_tag]
        elif self.task == 'tts':
            # Hardcode for tts online models.
            tts_online_models = [
                'fastspeech2_csmsc-zh', 'fastspeech2_cnndecoder_csmsc-zh',
                'mb_melgan_csmsc-zh', 'hifigan_csmsc-zh'
            ]
            for model_tag in list(self.pretrained_models.keys()):
                if inference_mode == 'online' and model_tag in tts_online_models:
                    continue
                elif inference_mode == 'offline':
                    continue
                else:
                    del self.pretrained_models[model_tag]
        else:
            raise NotImplementedError('Only supports asr and tts task.')

    @staticmethod
    def _fetch(res_dict: Dict[str, str],
               target_dir: os.PathLike) -> os.PathLike:
        """Fetch archive from url.

        Args:
            res_dict (Dict[str, str]): Info dict of a resource.
            target_dir (os.PathLike): Directory to save archives.

        Returns:
            os.PathLike: Directory of model resource.
        """
        return download_and_decompress(res_dict, target_dir)

    @staticmethod
    def _format_path(res_dict: Dict[str, str]):
        for k, v in res_dict.items():
            if isinstance(v, str) and '/' in v:
                if v.startswith('https://') or v.startswith('http://'):
                    continue
                else:
                    res_dict[k] = os.path.join(*(v.split('/')))
