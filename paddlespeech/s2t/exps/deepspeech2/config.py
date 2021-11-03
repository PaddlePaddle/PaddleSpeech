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
from yacs.config import CfgNode

from paddlespeech.s2t.exps.deepspeech2.model import DeepSpeech2Tester
from paddlespeech.s2t.exps.deepspeech2.model import DeepSpeech2Trainer
from paddlespeech.s2t.io.collator import SpeechCollator
from paddlespeech.s2t.io.dataset import ManifestDataset
from paddlespeech.s2t.models.ds2 import DeepSpeech2Model
from paddlespeech.s2t.models.ds2_online import DeepSpeech2ModelOnline


def get_cfg_defaults(model_type='offline'):
    _C = CfgNode()
    _C.data = ManifestDataset.params()
    _C.collator = SpeechCollator.params()
    _C.training = DeepSpeech2Trainer.params()
    _C.decoding = DeepSpeech2Tester.params()
    if model_type == 'offline':
        _C.model = DeepSpeech2Model.params()
    else:
        _C.model = DeepSpeech2ModelOnline.params()
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    config.set_new_allowed(True)
    return config
