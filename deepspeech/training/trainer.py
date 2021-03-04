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

import time
import logging
import logging.handlers
from pathlib import Path
import numpy as np
from collections import defaultdict

import paddle
from paddle import distributed as dist
from paddle.distributed.utils import get_gpus
from tensorboardX import SummaryWriter

from deepspeech.utils import checkpoint
from deepspeech.utils import mp_tools

__all__ = ["Trainer"]


class Trainer():
    """
    An experiment template in order to structure the training code and take 
    care of saving, loading, logging, visualization stuffs. It's intended to 
    be flexible and simple. 
    
    So it only handles output directory (create directory for the output, 
    create a checkpoint directory, dump the config in use and create 
    visualizer and logger) in a standard way without enforcing any
    input-output protocols to the model and dataloader. It leaves the main 
    part for the user to implement their own (setup the model, criterion, 
    optimizer, define a training step, define a validation function and 
    customize all the text and visual logs).
    It does not save too much boilerplate code. The users still have to write 
    the forward/backward/update mannually, but they are free to add 
    non-standard behaviors if needed.
    We have some conventions to follow.
    1. Experiment should have ``model``, ``optimizer``, ``train_loader`` and 
    ``valid_loader``, ``config`` and ``args`` attributes.
    2. The config should have a ``training`` field, which has 
    ``valid_interval``, ``save_interval`` and ``max_iteration`` keys. It is 
    used as the trigger to invoke validation, checkpointing and stop of the 
    experiment.
    3. There are four methods, namely ``train_batch``, ``valid``, 
    ``setup_model`` and ``setup_dataloader`` that should be implemented.
    Feel free to add/overwrite other methods and standalone functions if you 
    need.
    
    Parameters
    ----------
    config: yacs.config.CfgNode
        The configuration used for the experiment.
    
    args: argparse.Namespace
        The parsed command line arguments.
    Examples
    --------
    >>> def main_sp(config, args):
    >>>     exp = Trainer(config, args)
    >>>     exp.setup()
    >>>     exp.run()
    >>> 
    >>> config = get_cfg_defaults()
    >>> parser = default_argument_parser()
    >>> args = parser.parse_args()
    >>> if args.config: 
    >>>     config.merge_from_file(args.config)
    >>> if args.opts:
    >>>     config.merge_from_list(args.opts)
    >>> config.freeze()
    >>> 
    >>> if args.nprocs > 1 and args.device == "gpu":
    >>>     dist.spawn(main_sp, args=(config, args), nprocs=args.nprocs)
    >>> else:
    >>>     main_sp(config, args)
    """

    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.optimizer = None
        self.visualizer = None
        self.output_dir = None
        self.checkpoint_dir = None
        self.logger = None

    def setup(self):
        """Setup the experiment.
        """
        paddle.set_device(self.args.device)
        if self.parallel:
            self.init_parallel()

        self.setup_output_dir()
        self.dump_config()
        self.setup_visualizer()
        self.setup_logger()
        self.setup_checkpointer()

        self.setup_dataloader()
        self.setup_model()

        self.iteration = 0
        self.epoch = 0

    @property
    def parallel(self):
        """A flag indicating whether the experiment should run with 
        multiprocessing.
        """
        return self.args.device == "gpu" and self.args.nprocs > 1

    def init_parallel(self):
        """Init environment for multiprocess training.
        """
        dist.init_parallel_env()

    @mp_tools.rank_zero_only
    def save(self):
        """Save checkpoint (model parameters and optimizer states).
        """
        checkpoint.save_parameters(self.checkpoint_dir, self.iteration,
                                   self.model, self.optimizer)

    def resume_or_load(self):
        """Resume from latest checkpoint at checkpoints in the output 
        directory or load a specified checkpoint.
        
        If ``args.checkpoint_path`` is not None, load the checkpoint, else
        resume training.
        """
        iteration = checkpoint.load_parameters(
            self.model,
            self.optimizer,
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_path=self.args.checkpoint_path)
        self.iteration = iteration

    def new_epoch(self):
        """Reset the train loader and increment ``epoch``.
        """
        if self.parallel:
            # batch sampler epoch start from 0
            self.train_loader.batch_sampler.set_epoch(self.epoch)
        self.epoch += 1

    def train(self):
        """The training process.
        
        It includes forward/backward/update and periodical validation and 
        saving.
        """
        self.logger.info(
            f"Train Total Examples: {len(self.train_loader.dataset)}")
        self.new_epoch()
        while self.epoch <= self.config.training.n_epoch:
            try:
                for batch in self.train_loader:
                    self.iteration += 1
                    self.train_batch(batch)
            except Exception as e:
                self.logger.error(e)
                pass

            self.valid()
            self.save()
            self.lr_scheduler.step()
            self.new_epoch()

    def run(self):
        """The routine of the experiment after setup. This method is intended
        to be used by the user.
        """
        self.resume_or_load()
        try:
            self.train()
        except KeyboardInterrupt:
            self.save()
            exit(-1)
        finally:
            self.destory()

    def setup_output_dir(self):
        """Create a directory used for output.
        """
        # output dir
        output_dir = Path(self.args.output).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir = output_dir

    def setup_checkpointer(self):
        """Create a directory used to save checkpoints into.
        
        It is "checkpoints" inside the output directory.
        """
        # checkpoint dir
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        self.checkpoint_dir = checkpoint_dir

    @mp_tools.rank_zero_only
    def destory(self):
        # https://github.com/pytorch/fairseq/issues/2357
        if self.visualizer:
            self.visualizer.close()

    @mp_tools.rank_zero_only
    def setup_visualizer(self):
        """Initialize a visualizer to log the experiment.
        
        The visual log is saved in the output directory.
        
        Notes
        ------
        Only the main process has a visualizer with it. Use multiple 
        visualizers in multiprocess to write to a same log file may cause 
        unexpected behaviors.
        """
        # visualizer
        visualizer = SummaryWriter(logdir=str(self.output_dir))

        self.visualizer = visualizer

    def setup_logger(self):
        """Initialize a text logger to log the experiment.
        
        Each process has its own text logger. The logging message is write to 
        the standard output and a text file named ``worker_n.log`` in the 
        output directory, where ``n`` means the rank of the process. 
        when - how to split the log file by time interval
            'S' : Seconds
            'M' : Minutes
            'H' : Hours
            'D' : Days
            'W' : Week day
            default value: 'D'
        format - format of the log
            default format:
            %(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s
            INFO: 12-09 18:02:42: log.py:40 * 139814749787872 HELLO WORLD
        backup - how many backup file to keep
            default value: 7
        """
        when = 'D'
        backup = 7
        format = '[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s'
        formatter = logging.Formatter(fmt=format, datefmt='%Y/%m/%d %H:%M:%S')

        logger = logging.getLogger(__name__)
        logger.setLevel("INFO")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        log_file = self.output_dir / 'worker_{}.log'.format(dist.get_rank())
        # file_handler = logging.FileHandler(str(log_file))
        # file_handler.setFormatter(formatter)
        # logger.addHandler(file_handler)

        # handler = logging.handlers.TimedRotatingFileHandler(
        #     str(self.output_dir / "warning.log"), when=when, backupCount=backup)
        # handler.setLevel(logging.WARNING)
        # handler.setFormatter(formatter)
        # logger.addHandler(handler)

        # global logger
        stdout = False
        save_path = log_file
        logging.basicConfig(
            level=logging.DEBUG if stdout else logging.INFO,
            format=format,
            datefmt='%Y/%m/%d %H:%M:%S',
            filename=save_path if not stdout else None)
        self.logger = logger

    @mp_tools.rank_zero_only
    def dump_config(self):
        """Save the configuration used for this experiment. 
        
        It is saved in to ``config.yaml`` in the output directory at the 
        beginning of the experiment.
        """
        with open(self.output_dir / "config.yaml", 'wt') as f:
            print(self.config, file=f)

    def train_batch(self):
        """The training loop. A subclass should implement this method.
        """
        raise NotImplementedError("train_batch should be implemented.")

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def valid(self):
        """The validation. A subclass should implement this method.
        """
        raise NotImplementedError("valid should be implemented.")

    def setup_model(self):
        """Setup model, criterion and optimizer, etc. A subclass should 
        implement this method.
        """
        raise NotImplementedError("setup_model should be implemented.")

    def setup_dataloader(self):
        """Setup training dataloader and validation dataloader. A subclass 
        should implement this method.
        """
        raise NotImplementedError("setup_dataloader should be implemented.")
