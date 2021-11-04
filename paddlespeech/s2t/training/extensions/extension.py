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
# Modified from chainer(https://github.com/chainer/chainer)
PRIORITY_WRITER = 300
PRIORITY_EDITOR = 200
PRIORITY_READER = 100


class Extension():
    """Extension to customize the behavior of Trainer."""
    trigger = (1, 'iteration')
    priority = PRIORITY_READER
    name = None

    @property
    def default_name(self):
        """Default name of the extension, class name by default."""
        return type(self).__name__

    def __call__(self, trainer):
        """Main action of the extention. After each update, it is executed
        when the trigger fires."""
        raise NotImplementedError(
            'Extension implementation must override __call__.')

    def initialize(self, trainer):
        """Action that is executed once to get the corect trainer state.
        It is called before training normally, but if the trainer restores
        states with an Snapshot extension, this method should also be called.
        """
        pass

    def on_error(self, trainer, exc, tb):
        """Handles the error raised during training before finalization.
        """
        pass

    def finalize(self, trainer):
        """Action that is executed when training is done.
        For example, visualizers would need to be closed.
        """
        pass
