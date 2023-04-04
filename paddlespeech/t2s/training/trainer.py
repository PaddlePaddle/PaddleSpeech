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
import sys
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Callable
from typing import List
from typing import Union

import six

from paddlespeech.t2s.training.extension import Extension
from paddlespeech.t2s.training.extension import PRIORITY_READER
from paddlespeech.t2s.training.reporter import scope
from paddlespeech.t2s.training.trigger import get_trigger
from paddlespeech.t2s.training.triggers.limit_trigger import LimitTrigger
from paddlespeech.t2s.training.updater import UpdaterBase
from paddlespeech.t2s.utils import profiler


class _ExtensionEntry(object):
    def __init__(self, extension, trigger, priority):
        self.extension = extension
        self.trigger = trigger
        self.priority = priority


class Trainer(object):
    def __init__(self,
                 updater: UpdaterBase,
                 stop_trigger: Callable = None,
                 out: Union[str, Path] = 'result',
                 extensions: List[Extension] = None,
                 profiler_options: str = None):
        self.updater = updater
        self.extensions = OrderedDict()
        self.stop_trigger = LimitTrigger(*stop_trigger)
        self.out = Path(out)
        self.observation = None
        self.profiler_options = profiler_options
        self._done = False
        if extensions:
            for ext in extensions:
                self.extend(ext)

    @property
    def is_before_training(self):
        return self.updater.state.iteration == 0

    def extend(self, extension, name=None, trigger=None, priority=None):
        # get name for the extension
        # argument \
        # -> extention's name \
        # -> default_name (class name, when it is an object) \
        # -> function name when it is a function \
        # -> error

        if name is None:
            name = getattr(extension, 'name', None)
            if name is None:
                name = getattr(extension, 'default_name', None)
                if name is None:
                    name = getattr(extension, '__name__', None)
                    if name is None:
                        raise ValueError("Name is not given for the extension.")
        if name == 'training':
            raise ValueError("training is a reserved name.")

        if trigger is None:
            trigger = getattr(extension, 'trigger', (1, 'iteration'))
        trigger = get_trigger(trigger)

        if priority is None:
            priority = getattr(extension, 'priority', PRIORITY_READER)

        # add suffix to avoid nameing conflict
        ordinal = 0
        modified_name = name
        while modified_name in self.extensions:
            ordinal += 1
            modified_name = f"{name}_{ordinal}"
        extension.name = modified_name

        self.extensions[modified_name] = _ExtensionEntry(
            extension, trigger, priority)

    def get_extension(self, name):
        """get extension by name."""
        extensions = self.extensions
        if name in extensions:
            return extensions[name].extension
        else:
            raise ValueError(f'extension {name} not found')

    def run(self):
        if self._done:
            raise RuntimeError("Training is already done!.")

        self.out.mkdir(parents=True, exist_ok=True)

        # sort extensions by priorities once
        extension_order = sorted(
            self.extensions.keys(),
            key=lambda name: self.extensions[name].priority,
            reverse=True)
        extensions = [(name, self.extensions[name]) for name in extension_order]

        # initializing all extensions
        for name, entry in extensions:
            if hasattr(entry.extension, "initialize"):
                entry.extension.initialize(self)

        update = self.updater.update  # training step

        stop_trigger = self.stop_trigger

        # display only one progress bar
        max_iteration = None
        if isinstance(stop_trigger, LimitTrigger):
            if stop_trigger.unit == 'epoch':
                max_epoch = self.stop_trigger.limit
                updates_per_epoch = getattr(self.updater, "updates_per_epoch",
                                            None)
                max_iteration = max_epoch * updates_per_epoch if updates_per_epoch else None
            else:
                max_iteration = self.stop_trigger.limit

        try:
            while not stop_trigger(self):
                self.observation = {}
                # set observation as the report target
                # you can use report freely in Updater.update()

                # updating parameters and state
                with scope(self.observation):

                    update()
                    if self.profiler_options:
                        profiler.add_profiler_step(self.profiler_options)
                    batch_read_time = self.updater.batch_read_time
                    batch_time = self.updater.batch_time
                    avg_batch_cost = batch_read_time + batch_time
                    logger = self.updater.logger
                    logger.removeHandler(self.updater.filehandler)
                    msg = self.updater.msg
                    msg = " iter: {}/{}, ".format(self.updater.state.iteration,
                                                  max_iteration) + msg
                    msg += ", avg_reader_cost: {:.5f} sec, ".format(
                        batch_read_time
                    ) + "avg_batch_cost: {:.5f} sec, ".format(avg_batch_cost)
                    msg += "avg_samples: {}, ".format(
                        self.updater.batch_size
                    ) + "avg_ips: {:.5f} sequences/sec".format(
                        self.updater.batch_size / avg_batch_cost)
                    logger.info(msg)

                    # execute extension when necessary
                    for name, entry in extensions:
                        if entry.trigger(self):
                            entry.extension(self)

                # print("###", self.observation)
        except Exception as e:
            f = sys.stderr
            f.write(f"Exception in main training loop: {e}\n")
            f.write("Traceback (most recent call last):\n")
            traceback.print_tb(sys.exc_info()[2])
            f.write(
                "Trainer extensions will try to handle the extension. Then all extensions will finalize."
            )

            # capture the exception in the mian training loop
            exc_info = sys.exc_info()

            # try to handle it
            for name, entry in extensions:
                if hasattr(entry.extension, "on_error"):
                    try:
                        entry.extension.on_error(self, e, sys.exc_info()[2])
                    except Exception as ee:
                        f.write(f"Exception in error handler: {ee}\n")
                        f.write('Traceback (most recent call last):\n')
                        traceback.print_tb(sys.exc_info()[2])

            # raise exception in main training loop
            six.reraise(*exc_info)
        finally:
            for name, entry in extensions:
                if hasattr(entry.extension, "finalize"):
                    entry.extension.finalize(self)
