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
import logging
import os
import sys
from abc import ABC
from abc import abstractmethod
from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import paddle

from ..resource import CommonTaskResource
from .log import logger


class BaseExecutor(ABC):
    """
        An abstract executor of paddlespeech tasks.
    """

    def __init__(self, task: str, **kwargs):
        self._inputs = OrderedDict()
        self._outputs = OrderedDict()
        self.task_resource = CommonTaskResource(task=task, **kwargs)

    @abstractmethod
    def _init_from_path(self, *args, **kwargs):
        """
        Init model and other resources from arguments. This method should be called by `__call__()`.
        """
        pass

    @abstractmethod
    def preprocess(self, input: Any, *args, **kwargs):
        """
        Input preprocess and return paddle.Tensor stored in self._inputs.
        Input content can be a text(tts), a file(asr, cls), a stream(not supported yet) or anything needed.

        Args:
            input (Any): Input text/file/stream or other content.
        """
        pass

    @paddle.no_grad()
    @abstractmethod
    def infer(self, *args, **kwargs):
        """
        Model inference and put results into self._outputs.
        This method get input tensors from self._inputs, and write output tensors into self._outputs.
        """
        pass

    @abstractmethod
    def postprocess(self, *args, **kwargs) -> Union[str, os.PathLike]:
        """
        Output postprocess and return results.
        This method get model output from self._outputs and convert it into human-readable results.

        Returns:
            Union[str, os.PathLike]: Human-readable results such as texts and audio files.
        """
        pass

    @abstractmethod
    def execute(self, argv: List[str]) -> bool:
        """
        Command line entry. This method can only be accessed by a command line such as `paddlespeech asr`.

        Args:
            argv (List[str]): Arguments from command line.

        Returns:
            int: Result of the command execution. `True` for a success and `False` for a failure.
        """
        pass

    @abstractmethod
    def __call__(self, *arg, **kwargs):
        """
        Python API to call an executor.
        """
        pass

    def get_input_source(self, input_: Union[str, os.PathLike, None]
                         ) -> Dict[str, Union[str, os.PathLike]]:
        """
        Get task input source from command line input.

        Args:
            input_ (Union[str, os.PathLike, None]): Input from command line.

        Returns:
            Dict[str, Union[str, os.PathLike]]: A dict with ids and inputs.
        """
        if self._is_job_input(input_):
            ret = self._get_job_contents(input_)
        else:
            ret = OrderedDict()

            if input_ is None:  # Take input from stdin
                if not sys.stdin.isatty(
                ):  # Avoid getting stuck when stdin is empty.
                    for i, line in enumerate(sys.stdin):
                        line = line.strip()
                        if len(line.split(' ')) == 1:
                            ret[str(i + 1)] = line
                        elif len(line.split(' ')) == 2:
                            id_, info = line.split(' ')
                            ret[id_] = info
                        else:  # No valid input info from one line.
                            continue
            else:
                ret[1] = input_
        return ret

    def process_task_results(self,
                             input_: Union[str, os.PathLike, None],
                             results: Dict[str, os.PathLike],
                             job_dump_result: bool=False):
        """
        Handling task results and redirect stdout if needed.

        Args:
            input_ (Union[str, os.PathLike, None]): Input from command line.
            results (Dict[str, os.PathLike]): Task outputs.
            job_dump_result (bool, optional): if True, dumps job results into file. Defaults to False.
        """

        if not self._is_job_input(input_) and len(
                results) == 1:  # Only one input sample
            raw_text = list(results.values())[0]
        else:
            raw_text = self._format_task_results(results)

        print(raw_text, end='')  # Stdout

        if self._is_job_input(
                input_) and job_dump_result:  # Dump to *.job.done 
            try:
                job_output_file = os.path.abspath(input_) + '.done'
                sys.stdout = open(job_output_file, 'w')
                print(raw_text, end='')
                logger.info(f'Results had been saved to: {job_output_file}')
            finally:
                sys.stdout.close()

    def _is_job_input(self, input_: Union[str, os.PathLike]) -> bool:
        """
        Check if current input file is a job input or not.

        Args:
            input_ (Union[str, os.PathLike]): Input file of current task.

        Returns:
            bool: return `True` for job input, `False` otherwise.
        """
        return input_ and os.path.isfile(input_) and (input_.endswith('.job') or
                                                      input_.endswith('.txt'))

    def _get_job_contents(
            self, job_input: os.PathLike) -> Dict[str, Union[str, os.PathLike]]:
        """
        Read a job input file and return its contents in a dictionary.

        Args:
            job_input (os.PathLike): The job input file.

        Returns:
            Dict[str, str]: Contents of job input.
        """
        job_contents = OrderedDict()
        with open(job_input) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                k, v = line.split(' ')
                job_contents[k] = v
        return job_contents

    def _format_task_results(
            self, results: Dict[str, Union[str, os.PathLike]]) -> str:
        """
        Convert task results to raw text.

        Args:
            results (Dict[str, str]): A dictionary of task results.

        Returns:
            str: A string object contains task results.
        """
        ret = ''
        for k, v in results.items():
            ret += f'{k} {v}\n'
        return ret

    def disable_task_loggers(self):
        """
        Disable all loggers in current task.
        """
        loggers = [
            logging.getLogger(name) for name in logging.root.manager.loggerDict
        ]
        for l in loggers:
            l.setLevel(logging.ERROR)

    def show_rtf(self, info: Dict[str, List[float]]):
        """
        Calculate rft of current task and show results.
        """
        num_samples = 0
        task_duration = 0.0
        wav_duration = 0.0

        for start, end, dur in zip(info['start'], info['end'], info['extra']):
            num_samples += 1
            task_duration += end - start
            wav_duration += dur

        logger.info('Sample Count: {}'.format(num_samples))
        logger.info('Avg RTF: {}'.format(task_duration / wav_duration))
