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
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import paddle
from paddle import distributed as dist
from visualdl import LogWriter

from paddlespeech.s2t.training.reporter import ObsScope
from paddlespeech.s2t.training.reporter import report
from paddlespeech.s2t.training.timer import Timer
from paddlespeech.s2t.utils import mp_tools
from paddlespeech.s2t.utils import profiler
from paddlespeech.s2t.utils.checkpoint import Checkpoint
from paddlespeech.s2t.utils.log import Log
from paddlespeech.s2t.utils.utility import all_version
from paddlespeech.s2t.utils.utility import seed_all
from paddlespeech.s2t.utils.utility import UpdateConfig

__all__ = ["Trainer"]

logger = Log(__name__).getlog()


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
    >>> if args.ngpu > 1:
    >>>     dist.spawn(main_sp, args=(config, args), nprocs=args.ngpu)
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
        self.iteration = 0
        self.epoch = 0
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._train = True

        # print deps version
        all_version()
        logger.info(f"Rank: {self.rank}/{self.world_size}")

        # set device
        if self.args.ngpu == 0:
            paddle.set_device('cpu')
            if self.args.nxpu == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('xpu')
        elif self.args.ngpu > 0:
            paddle.set_device("gpu")
        if self.parallel:
            self.init_parallel()

        self.checkpoint = Checkpoint(
            kbest_n=self.config.checkpoint.kbest_n,
            latest_n=self.config.checkpoint.latest_n)

        # set random seed if needed
        if args.seed:
            seed_all(args.seed)
            logger.info(f"Set seed {args.seed}")

        # profiler and benchmark options
        if hasattr(self.args,
                   "benchmark_batch_size") and self.args.benchmark_batch_size:
            with UpdateConfig(self.config):
                self.config.batch_size = self.args.benchmark_batch_size
                self.config.log_interval = 1
            logger.info(
                f"Benchmark reset batch-size: {self.args.benchmark_batch_size}")

    @property
    def train(self):
        return self._train

    @contextmanager
    def eval(self):
        self._train = False
        yield
        self._train = True

    def setup(self):
        """Setup the experiment.
        """
        self.setup_output_dir()
        self.dump_config()
        self.setup_visualizer()

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
    def save(self, tag=None, infos: dict=None):
        """Save checkpoint (model parameters and optimizer states).

        Args:
            tag (int or str, optional): None for step, else using tag, e.g epoch. Defaults to None.
            infos (dict, optional): meta data to save. Defaults to None.
        """

        infos = infos if infos else dict()
        infos.update({
            "step": self.iteration,
            "epoch": self.epoch,
            "lr": self.optimizer.get_lr()
        })
        self.checkpoint.save_parameters(self.checkpoint_dir, self.iteration
                                        if tag is None else tag, self.model,
                                        self.optimizer, infos)

    def resume_or_scratch(self):
        """Resume from latest checkpoint at checkpoints in the output
        directory or load a specified checkpoint.

        If ``args.checkpoint_path`` is not None, load the checkpoint, else
        resume training.
        """
        scratch = None
        infos = self.checkpoint.load_latest_parameters(
            self.model,
            self.optimizer,
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_path=self.args.checkpoint_path)
        if infos:
            # just restore ckpt
            # lr will resotre from optimizer ckpt
            self.iteration = infos["step"]
            self.epoch = infos["epoch"]
            scratch = False
            logger.info(
                f"Restore ckpt: epoch {self.epoch }, step {self.iteration}!")
        else:
            self.iteration = 0
            self.epoch = 0
            scratch = True
            logger.info("Init from scratch!")
        return scratch

    def maybe_batch_sampler_step(self):
        """ batch_sampler seed by epoch """
        if hasattr(self.train_loader, "batch_sampler"):
            batch_sampler = self.train_loader.batch_sampler
            if isinstance(batch_sampler, paddle.io.DistributedBatchSampler):
                logger.debug(
                    f"train_loader.batch_sample.set_epoch: {self.epoch}")
                batch_sampler.set_epoch(self.epoch)

    def before_train(self):
        from_scratch = self.resume_or_scratch()
        if from_scratch:
            # scratch: save init model, i.e. 0 epoch
            self.save(tag='init', infos=None)
        else:
            # resume: train next_epoch and next_iteration
            self.epoch += 1
            self.iteration += 1
            logger.info(
                f"Resume train: epoch {self.epoch }, step {self.iteration}!")

        self.maybe_batch_sampler_step()

    def new_epoch(self):
        """Reset the train loader seed and increment `epoch`.
        """
        # `iteration` increased by train step
        self.epoch += 1
        self.maybe_batch_sampler_step()

    def after_train_batch(self):
        if self.args.benchmark_max_step:
            profiler.add_profiler_step(self.args.profiler_options)
        if self.args.benchmark_max_step and self.iteration > self.args.benchmark_max_step:
            logger.info(
                f"Reach benchmark-max-step: {self.args.benchmark_max_step}")
            sys.exit(0)

    def do_train(self):
        """The training process control by epoch."""
        self.before_train()

        logger.info(f"Train Total Examples: {len(self.train_loader.dataset)}")
        while self.epoch < self.config.n_epoch:
            with Timer("Epoch-Train Time Cost: {}"):
                self.model.train()
                try:
                    data_start_time = time.time()
                    for batch_index, batch in enumerate(self.train_loader):
                        dataload_time = time.time() - data_start_time
                        msg = "Train:"
                        observation = OrderedDict()
                        with ObsScope(observation):
                            report("Rank", dist.get_rank())
                            report("epoch", self.epoch)
                            report('step', self.iteration)
                            report("lr", self.lr_scheduler())
                            self.train_batch(batch_index, batch, msg)
                            self.after_train_batch()
                            report('iter', batch_index + 1)
                            report('total', len(self.train_loader))
                            report('reader_cost', dataload_time)
                        observation['batch_cost'] = observation[
                            'reader_cost'] + observation['step_cost']
                        observation['samples'] = observation['batch_size']
                        observation['ips samples/s'] = observation[
                            'batch_size'] / observation['batch_cost']
                        for k, v in observation.items():
                            msg += f" {k}: "
                            msg += f"{v:>.8f}" if isinstance(v,
                                                             float) else f"{v}"
                            msg += ","
                        msg = msg[:-1]  # remove the last ","
                        if (batch_index + 1) % self.config.log_interval == 0:
                            logger.info(msg)
                        data_start_time = time.time()
                except Exception as e:
                    logger.error(e)
                    raise e

            with Timer("Eval Time Cost: {}"):
                total_loss, num_seen_utts = self.valid()
                if dist.get_world_size() > 1:
                    num_seen_utts = paddle.to_tensor(num_seen_utts)
                    # the default operator in all_reduce function is sum.
                    dist.all_reduce(num_seen_utts)
                    total_loss = paddle.to_tensor(total_loss)
                    dist.all_reduce(total_loss)
                    cv_loss = total_loss / num_seen_utts
                    cv_loss = float(cv_loss)
                else:
                    cv_loss = total_loss / num_seen_utts

            logger.info(
                'Epoch {} Val info val_loss {}'.format(self.epoch, cv_loss))
            if self.visualizer:
                self.visualizer.add_scalar(
                    tag='eval/cv_loss', value=cv_loss, step=self.epoch)
                self.visualizer.add_scalar(
                    tag='eval/lr', value=self.lr_scheduler(), step=self.epoch)

            # step lr every epoch
            self.lr_scheduler.step()
            # after epoch
            self.save(tag=self.epoch, infos={'val_loss': cv_loss})
            self.new_epoch()

    def run(self):
        """The routine of the experiment after setup. This method is intended
        to be used by the user.
        """
        try:
            with Timer("Training Done: {}"):
                self.do_train()
        except KeyboardInterrupt:
            exit(-1)
        finally:
            self.destory()

    def restore(self):
        """Resume from latest checkpoint at checkpoints in the output
        directory or load a specified checkpoint.

        If ``args.checkpoint_path`` is not None, load the checkpoint, else
        resume training.
        """
        assert self.args.checkpoint_path
        infos = self.checkpoint.load_latest_parameters(
            self.model, checkpoint_path=self.args.checkpoint_path)
        return infos

    def run_test(self):
        """Do Test/Decode"""
        try:
            with Timer("Test/Decode Done: {}"):
                with self.eval():
                    self.restore()
                    self.test()
        except KeyboardInterrupt:
            exit(-1)

    def run_export(self):
        """Do Model Export"""
        try:
            with Timer("Export Done: {}"):
                with self.eval():
                    self.restore()
                    self.export()
        except KeyboardInterrupt:
            exit(-1)

    def run_align(self):
        """Do CTC alignment"""
        try:
            with Timer("Align Done: {}"):
                with self.eval():
                    self.restore()
                    self.align()
        except KeyboardInterrupt:
            sys.exit(-1)

    def setup_output_dir(self):
        """Create a directory used for output.
        """
        if self.args.output:
            output_dir = Path(self.args.output).expanduser()
        elif self.args.checkpoint_path:
            output_dir = Path(
                self.args.checkpoint_path).expanduser().parent.parent
        elif self.args.export_path:
            output_dir = Path(self.args.export_path).expanduser().parent.parent
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = output_dir / "log"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.test_dir = output_dir / "test"
        self.test_dir.mkdir(parents=True, exist_ok=True)

        self.decode_dir = output_dir / "decode"
        self.decode_dir.mkdir(parents=True, exist_ok=True)

        self.export_dir = output_dir / "export"
        self.export_dir.mkdir(parents=True, exist_ok=True)

        self.visual_dir = output_dir / "visual"
        self.visual_dir.mkdir(parents=True, exist_ok=True)

        self.config_dir = output_dir / "conf"
        self.config_dir.mkdir(parents=True, exist_ok=True)

    @mp_tools.rank_zero_only
    def destory(self):
        """Close visualizer to avoid hanging after training"""
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
        visualizer = LogWriter(logdir=str(self.visual_dir))
        self.visualizer = visualizer

    @mp_tools.rank_zero_only
    def dump_config(self):
        """Save the configuration used for this experiment.

        It is saved in to ``config.yaml`` in the output directory at the
        beginning of the experiment.
        """
        config_file = self.config_dir / "config.yaml"
        if self.train and config_file.exists():
            time_stamp = time.strftime("%Y_%m_%d_%H_%M_%s", time.gmtime())
            target_path = self.config_dir / ".".join(
                [time_stamp, "config.yaml"])
            config_file.rename(target_path)

        with open(config_file, 'wt') as f:
            print(self.config, file=f)

    def train_batch(self):
        """The training loop. A subclass should implement this method.
        """
        raise NotImplementedError("train_batch should be implemented.")

    @paddle.no_grad()
    def valid(self):
        """The validation. A subclass should implement this method.
        """
        raise NotImplementedError("valid should be implemented.")

    @paddle.no_grad()
    def test(self):
        """The test. A subclass should implement this method in Tester.
        """
        raise NotImplementedError("test should be implemented.")

    @paddle.no_grad()
    def export(self):
        """The test. A subclass should implement this method in Tester.
        """
        raise NotImplementedError("export should be implemented.")

    @paddle.no_grad()
    def align(self):
        """The align. A subclass should implement this method in Tester.
        """
        raise NotImplementedError("align should be implemented.")

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
