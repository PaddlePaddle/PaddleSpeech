from typing import Dict
from typing import Optional

from paddle import Tensor
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler
from paddle.nn import Layer
from paddle.optimizer import Optimizer
from timer import timer

from deepspeech.training.reporter import report
from deepspeech.training.updaters.updater import UpdaterBase
from deepspeech.training.updaters.updater import UpdaterState

from deepspeech.utils.log import Log

__all__ = ["StandardUpdater"]

logger = Log(__name__).getlog()

class StandardUpdater(UpdaterBase):
    """An example of over-simplification. Things may not be that simple, but
    you can subclass it to fit your need.
    """

    def __init__(self,
                 model: Layer,
                 optimizer: Optimizer,
                 dataloader: DataLoader,
                 init_state: Optional[UpdaterState]=None):
        # it is designed to hold multiple models
        models = {"main": model}
        self.models: Dict[str, Layer] = models
        self.model = model

        # it is designed to hold multiple optimizers
        optimizers = {"main": optimizer}
        self.optimizer = optimizer
        self.optimizers: Dict[str, Optimizer] = optimizers

        # dataloaders
        self.dataloader = dataloader

        # init state
        if init_state is None:
            self.state = UpdaterState()
        else:
            self.state = init_state

        self.train_iterator = iter(dataloader)

    def update(self):
        # We increase the iteration index after updating and before extension.
        # Here are the reasons.

        # 0. Snapshotting(as well as other extensions, like visualizer) is
        #    executed after a step of updating;
        # 1. We decide to increase the iteration index after updating and
        #    before any all extension is executed. 
        # 3. We do not increase the iteration after extension because we
        #    prefer a consistent resume behavior, when load from a
        #    `snapshot_iter_100.pdz` then the next step to train is `101`,
        #    naturally. But if iteration is increased increased after
        #    extension(including snapshot), then, a `snapshot_iter_99` is
        #    loaded. You would need a extra increasing of the iteration idex
        #    before training to avoid another iteration `99`, which has been
        #    done before snapshotting.
        # 4. Thus iteration index represrnts "currently how mant epochs has
        #    been done."
        # NOTE: use report to capture the correctly value. If you want to
        # report the learning rate used for a step, you must report it before
        # the learning rate scheduler's step() has been called. In paddle's
        # convention, we do not use an extension to change the learning rate.
        # so if you want to report it, do it in the updater.

        # Then here comes the next question. When is the proper time to
        # increase the epoch index? Since all extensions are executed after
        # updating, it is the time that after updating is the proper time to
        # increase epoch index.
        # 1. If we increase the epoch index before updating, then an extension
        #    based ot epoch would miss the correct timing. It could only be
        #    triggerd after an extra updating.
        # 2. Theoretically, when an epoch is done, the epoch index should be
        #    increased. So it would be increase after updating.
        # 3. Thus, eppoch index represents "currently how many epochs has been
        #    done." So it starts from 0.

        # switch to training mode
        for model in self.models.values():
            model.train()

        # training for a step is implemented here
        batch = self.read_batch()
        self.update_core(batch)

        self.state.iteration += 1
        if self.updates_per_epoch is not None:
            if self.state.iteration % self.updates_per_epoch == 0:
                self.state.epoch += 1

    def update_core(self, batch):
        """A simple case for a training step. Basic assumptions are:
        Single model;
        Single optimizer;
        A batch from the dataloader is just the input of the model;
        The model return a single loss, or a dict containing serval losses.
        Parameters updates at every batch, no gradient accumulation.
        """
        loss = self.model(*batch)

        if isinstance(loss, Tensor):
            loss_dict = {"main": loss}
        else:
            # Dict[str, Tensor]
            loss_dict = loss
            if "main" not in loss_dict:
                main_loss = 0
                for loss_item in loss.values():
                    main_loss += loss_item
                loss_dict["main"] = main_loss

        for name, loss_item in loss_dict.items():
            report(name, float(loss_item))

        self.optimizer.clear_gradient()
        loss_dict["main"].backward()
        self.optimizer.update()

    @property
    def updates_per_epoch(self):
        """Number of updater per epoch, determined by the length of the
        dataloader."""
        length_of_dataloader = None
        try:
            length_of_dataloader = len(self.dataloader)
        except TypeError:
            logger.debug("This dataloader has no __len__.")
        finally:
            return length_of_dataloader

    def new_epoch(self):
        """Start a new epoch."""
        # NOTE: all batch sampler for distributed training should
        # subclass DistributedBatchSampler and implement `set_epoch` method
        if hasattr(self.dataloader, "batch_sampler")
            batch_sampler = self.dataloader.batch_sampler
            if isinstance(batch_sampler, DistributedBatchSampler):
                batch_sampler.set_epoch(self.state.epoch)
        self.train_iterator = iter(self.dataloader)

    def read_batch(self):
        """Read a batch from the data loader, auto renew when data is exhausted."""
        with timer() as t:
            try:
                batch = next(self.train_iterator)
            except StopIteration:
                self.new_epoch()
                batch = next(self.train_iterator)
            logger.debug(
                f"Read a batch takes {t.elapse}s.")  # replace it with logger
        return batch

    def state_dict(self):
        """State dict of a Updater, model, optimizer and updater state are included."""
        state_dict = super().state_dict()
        for name, model in self.models.items():
            state_dict[f"{name}_params"] = model.state_dict()
        for name, optim in self.optimizers.items():
            state_dict[f"{name}_optimizer"] = optim.state_dict()
        return state_dict

    def set_state_dict(self, state_dict):
        """Set state dict for a Updater. Parameters of models, states for
        optimizers and UpdaterState are restored."""
        for name, model in self.models.items():
            model.set_state_dict(state_dict[f"{name}_params"])
        for name, optim in self.optimizers.items():
            optim.set_state_dict(state_dict[f"{name}_optimizer"])
        super().set_state_dict(state_dict)