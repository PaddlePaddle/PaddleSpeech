# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import sys
from pathlib import Path

import paddle
from paddle import distributed as dist
from paddle.io import DistributedBatchSampler
from visualdl import LogWriter

from paddlespeech.t2s.utils import checkpoint
from paddlespeech.t2s.utils import mp_tools

__all__ = ["ExperimentBase"]


class ExperimentBase(object):
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
    >>>     exp = Experiment(config, args)
    >>>     exp.setup()
    >>>     exe.resume_or_load()
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
    >>> if args.ngpu > 1:
    >>>     dist.spawn(main_sp, args=(config, args), nprocs=args.ngpu)
    >>> else:
    >>>     main_sp(config, args)
    """

    def __init__(self, config, args):
        self.config = config
        self.args = args

        self.model = None
        self.optimizer = None
        self.iteration = 0
        self.epoch = 0
        self.train_loader = None
        self.valid_loader = None
        self.iterator = None
        self.logger = None
        self.visualizer = None
        self.output_dir = None
        self.checkpoint_dir = None

    def setup(self):
        """Setup the experiment.
        """
        if self.args.ngpu == 0:
            paddle.set_device("cpu")
        elif self.args.ngpu > 0:
            paddle.set_device("gpu")
        else:
            print("ngpu should >= 0 !")
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
        return self.args.ngpu > 1

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

    def read_batch(self):
        """Read a batch from the train_loader.

        Returns
        -------
        List[Tensor]
            A batch.
        """
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.new_epoch()
            batch = next(self.iterator)
        return batch

    def new_epoch(self):
        """Reset the train loader and increment ``epoch``.
        """
        self.epoch += 1
        if self.parallel and isinstance(self.train_loader.batch_sampler,
                                        DistributedBatchSampler):
            self.train_loader.batch_sampler.set_epoch(self.epoch)
        self.iterator = iter(self.train_loader)

    def train(self):
        """The training process.

        It includes forward/backward/update and periodical validation and
        saving.
        """
        self.new_epoch()
        while self.iteration < self.config.training.max_iteration:
            self.iteration += 1
            self.train_batch()

            if self.iteration % self.config.training.valid_interval == 0:
                self.valid()

            if self.iteration % self.config.training.save_interval == 0:
                self.save()

    def run(self):
        """The routine of the experiment after setup. This method is intended
        to be used by the user.
        """
        try:
            self.train()
        except KeyboardInterrupt as exception:
            # delete this, because it can not save a complete model
            # self.save()
            self.close()
            sys.exit(exception)
        finally:
            self.close()

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
    def close(self):
        """Close visualizer to avoid hanging after training"""
        # https://github.com/pytorch/fairseq/issues/2357
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
        visualizer = LogWriter(logdir=str(self.output_dir))

        self.visualizer = visualizer

    def setup_logger(self):
        """Initialize a text logger to log the experiment.

        Each process has its own text logger. The logging message is write to
        the standard output and a text file named ``worker_n.log`` in the
        output directory, where ``n`` means the rank of the process.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel("INFO")
        log_file = self.output_dir / 'worker_{}.log'.format(dist.get_rank())
        logger.addHandler(logging.FileHandler(str(log_file)))

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
