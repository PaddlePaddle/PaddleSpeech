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
from paddlespeech.s2t.training.trainer import Trainer
from paddlespeech.s2t.utils.dynamic_import import dynamic_import

model_trainer_alias = {
    "ds2": "paddlespeech.s2t.exp.deepspeech2.model:DeepSpeech2Trainer",
    "u2": "paddlespeech.s2t.exps.u2.model:U2Trainer",
    "u2_kaldi": "paddlespeech.s2t.exps.u2_kaldi.model:U2Trainer",
    "u2_st": "paddlespeech.s2t.exps.u2_st.model:U2STTrainer",
}


def dynamic_import_trainer(module):
    """Import Trainer dynamically.

    Args:
        module (str): trainer name. e.g., ds2, u2, u2_kaldi

    Returns:
        type: Trainer class

    """
    model_class = dynamic_import(module, model_trainer_alias)
    assert issubclass(model_class,
                      Trainer), f"{module} does not implement Trainer"
    return model_class


model_tester_alias = {
    "ds2": "paddlespeech.s2t.exp.deepspeech2.model:DeepSpeech2Tester",
    "u2": "paddlespeech.s2t.exps.u2.model:U2Tester",
    "u2_kaldi": "paddlespeech.s2t.exps.u2_kaldi.model:U2Tester",
    "u2_st": "paddlespeech.s2t.exps.u2_st.model:U2STTester",
}


def dynamic_import_tester(module):
    """Import Tester dynamically.

    Args:
        module (str): tester name. e.g., ds2, u2, u2_kaldi

    Returns:
        type: Tester class

    """
    model_class = dynamic_import(module, model_tester_alias)
    assert issubclass(model_class,
                      Trainer), f"{module} does not implement Tester"
    return model_class
