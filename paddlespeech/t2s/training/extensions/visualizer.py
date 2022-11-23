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
from visualdl import LogWriter

from paddlespeech.t2s.training import extension
from paddlespeech.t2s.training.trainer import Trainer


class VisualDL(extension.Extension):
    """A wrapper of visualdl log writer. It assumes that the metrics to be visualized
    are all scalars which are recorded into the `.observation` dictionary of the
    trainer object. The dictionary is created for each step, thus the visualdl log
    writer uses the iteration from the updater's `iteration` as the global step to
    add records.
    """
    trigger = (1, 'iteration')
    default_name = 'visualdl'
    priority = extension.PRIORITY_READER

    def __init__(self, logdir):
        self.writer = LogWriter(str(logdir))

    def __call__(self, trainer: Trainer):
        for k, v in trainer.observation.items():
            self.writer.add_scalar(k, v, step=trainer.updater.state.iteration)

    def finalize(self, trainer):
        self.writer.close()
