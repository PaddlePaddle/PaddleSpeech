"""
Schedulers for updating hyperparameters (such as learning rate).

Authors
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
 * Loren Lugosch 2020
"""

import math
import paddle
import logging
from speechbrain.utils import checkpoints

logger = logging.getLogger(__name__)


def update_learning_rate(optimizer, new_lr, param_group=None):
    """Change the learning rate value within an optimizer.

    Arguments
    ---------
    optimizer : torch.optim object
        Updates the learning rate for this optimizer.
    new_lr : float
        The new value to use for the learning rate.
    param_group : list of int
        The param group indices to update. If not provided, all groups updated.

    Example
    -------
    >>> from torch.optim import SGD
    >>> from speechbrain.nnet.linear import Linear
    >>> model = Linear(n_neurons=10, input_size=10)
    >>> optimizer = SGD(model.parameters(), lr=0.1)
    >>> update_learning_rate(optimizer, 0.2)
    >>> optimizer.param_groups[0]["lr"]
    0.2
    """
    # Iterate all groups if none is provided
    # if param_group is None:
    #     groups = range(len(optimizer.param_groups))
    old_lr = optimizer.get_lr()
    optimizer.set_lr(new_lr)
    logger.info("Changing lr from %.2g to %.2g" % (old_lr, new_lr))
    # for i in groups:
    #     old_lr = optimizer.param_groups[i]["lr"]

    #     # Change learning rate if new value is different from old.
    #     if new_lr != old_lr:
    #         optimizer.param_groups[i]["lr"] = new_lr
    #         optimizer.param_groups[i]["prev_lr"] = old_lr
    #         logger.info("Changing lr from %.2g to %.2g" % (old_lr, new_lr))


@checkpoints.register_checkpoint_hooks
class NewBobScheduler:
    """Scheduler with new-bob technique, used for LR annealing.

    The learning rate is annealed based on the validation performance.
    In particular: if (past_loss-current_loss)/past_loss< impr_threshold:
    lr=lr * annealing_factor.

    Arguments
    ---------
    initial_value : float
        The initial hyperparameter value.
    annealing_factor : float
        It is annealing factor used in new_bob strategy.
    improvement_threshold : float
        It is the improvement rate between losses used to perform learning
        annealing in new_bob strategy.
    patient : int
        When the annealing condition is violated patient times,
        the learning rate is finally reduced.

    Example
    -------
    >>> scheduler = NewBobScheduler(initial_value=1.0)
    >>> scheduler(metric_value=10.0)
    (1.0, 1.0)
    >>> scheduler(metric_value=2.0)
    (1.0, 1.0)
    >>> scheduler(metric_value=2.5)
    (1.0, 0.5)
    """

    def __init__(
        self,
        initial_value,
        annealing_factor=0.5,
        improvement_threshold=0.0025,
        patient=0,
    ):
        self.hyperparam_value = initial_value
        self.annealing_factor = annealing_factor
        self.improvement_threshold = improvement_threshold
        self.patient = patient
        self.metric_values = []
        self.current_patient = self.patient

    def __call__(self, metric_value):
        """Returns the current and new value for the hyperparameter.

        Arguments
        ---------
        metric_value : int
            A number for determining whether to change the hyperparameter value.
        """
        old_value = new_value = self.hyperparam_value
        if len(self.metric_values) > 0:
            prev_metric = self.metric_values[-1]
            # Update value if improvement too small and patience is 0
            if prev_metric == 0:  # Prevent division by zero
                improvement = 0
            else:
                improvement = (prev_metric - metric_value) / prev_metric
            if improvement < self.improvement_threshold:
                if self.current_patient == 0:
                    new_value *= self.annealing_factor
                    self.current_patient = self.patient
                else:
                    self.current_patient -= 1

        # Store relevant info
        self.metric_values.append(metric_value)
        self.hyperparam_value = new_value

        return old_value, new_value

    @checkpoints.mark_as_saver
    def save(self, path):
        data = {
            "hyperparam_value": self.hyperparam_value,
            "metric_values": self.metric_values,
            "current_patient": self.current_patient,
        }
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch=False, device=None):
        del end_of_epoch  # Unused in this class
        del device  # Unused in here
        data = torch.load(path)
        self.hyperparam_value = data["hyperparam_value"]
        self.metric_values = data["metric_values"]
        self.current_patient = data["current_patient"]


class LinearScheduler:
    """Scheduler with linear annealing technique.

    The learning rate linearly decays over the specified number of epochs.

    Arguments
    ---------
    initial_value : float
        The value upon initialization.
    final_value : float
        The value used when the epoch count reaches ``epoch_count - 1``.
    epoch_count : int
        Number of epochs.

    Example
    -------
    >>> scheduler = LinearScheduler(1.0, 0.0, 4)
    >>> scheduler(current_epoch=1)
    (1.0, 0.666...)
    >>> scheduler(current_epoch=2)
    (0.666..., 0.333...)
    >>> scheduler(current_epoch=3)
    (0.333..., 0.0)
    >>> scheduler(current_epoch=4)
    (0.0, 0.0)
    """

    def __init__(self, initial_value, final_value, epoch_count):
        self.value_at_epoch = paddle.linspace(
            initial_value, final_value, steps=epoch_count
        ).tolist()

    def __call__(self, current_epoch):
        """Returns the current and new value for the hyperparameter.

        Arguments
        ---------
        current_epoch : int
            Number of times the dataset has been iterated.
        """
        old_index = max(0, current_epoch - 1)
        index = min(current_epoch, len(self.value_at_epoch) - 1)
        return self.value_at_epoch[old_index], self.value_at_epoch[index]


class StepScheduler:
    """Learning rate scheduler with step annealing technique.

    The hyperparameter's value decays over the epochs with the
    selected ``epoch_decay`` factor.

    ``value = init_value * decay_factor ^ floor((1 + epoch) / decay_drop)``

    Arguments
    ---------
    initial_value : float
        Initial value for the hyperparameter being updated.
    decay_factor : float
        Factor multiplied with the initial_value
    decay_drop : float
        Annealing factor (the decay of the hyperparameter value is faster
        with higher ``decay_drop`` values).

    Example
    -------
    >>> scheduler = StepScheduler(initial_value=1.0)
    >>> scheduler(current_epoch=1)
    (1.0, 0.5)
    >>> scheduler(current_epoch=2)
    (0.5, 0.5)
    >>> scheduler(current_epoch=3)
    (0.5, 0.25)
    """

    def __init__(
        self, initial_value, decay_factor=0.5, decay_drop=2,
    ):
        self.initial_value = initial_value
        self.decay_factor = decay_factor
        self.decay_drop = decay_drop

    def __call__(self, current_epoch):
        """Returns current and new hyperparameter value.

        Arguments
        ---------
        current_epoch : int
            Number of times the dataset has been iterated.
        """
        current_value = self._compute_value(current_epoch - 1)
        next_value = self._compute_value(current_epoch)

        return current_value, next_value

    def _compute_value(self, current_epoch):
        return self.initial_value * math.pow(
            self.decay_factor,
            math.floor((1 + current_epoch) / self.decay_drop),
        )


@checkpoints.register_checkpoint_hooks
class NoamScheduler:
    """The is an implementation of the transformer's learning rate scheduler with warmup.
    Reference: https://arxiv.org/abs/1706.03762

    Note: this scheduler anneals the lr at each update of the model's weight,
    and n_steps must be saved for restarting.

    Arguments
    ---------
    lr_initial : float
        Initial learning rate (i.e. the lr used at epoch 0).
    n_warmup_steps : int
        numer of warm-up steps
    model_size : int
        size of transformer embed_dim. It is used to scale the maximum learning rate value reached
        by the scheduler. It is divided by model_size ** (0.5).
        If not specified the maximum learning rate value is instead multiplied by warmup_steps ** (0.5).

    Example
    -------
    >>> from speechbrain.nnet.linear import Linear
    >>> inp_tensor = torch.rand([1,660,3])
    >>> model = Linear(input_size=3, n_neurons=4)
    >>> optim = torch.optim.Adam(model.parameters(), lr=1)
    >>> output = model(inp_tensor)
    >>> scheduler =NoamScheduler(optim.param_groups[0]["lr"], 3)
    >>> curr_lr,next_lr=scheduler(optim)
    >>> optim.param_groups[0]["lr"]
    0.3333333333333333
    >>> curr_lr,next_lr=scheduler(optim)
    >>> optim.param_groups[0]["lr"]
    0.6666666666666666
    >>> curr_lr,next_lr=scheduler(optim)
    >>> optim.param_groups[0]["lr"]
    0.9999999999999999
    """

    def __init__(self, lr_initial, n_warmup_steps, model_size=None):
        self.lr_initial = lr_initial
        self.n_warmup_steps = n_warmup_steps
        self.current_lr = lr_initial
        self.losses = []
        self.n_steps = 0
        self.normalize = n_warmup_steps ** 0.5
        if model_size is not None:
            self.normalize = model_size ** (-0.5)

    def __call__(self, opt):
        """
        Arguments
        ---------
        opt : optimizer
            The optimizer to update using this scheduler.

        Returns
        -------
        current_lr : float
            The learning rate before the update.
        lr : float
            The learning rate after the update.
        """
        self.n_steps += 1

        current_lr = opt.param_groups[0]["lr"]

        lr = self.lr_initial * self._get_lr_scale()

        # Changing the learning rate within the optimizer
        for param_group in opt.param_groups:
            param_group["lr"] = lr

        self.current_lr = current_lr
        return current_lr, lr

    def _get_lr_scale(self):
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return self.normalize * min(
            n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5)
        )

    @checkpoints.mark_as_saver
    def save(self, path):
        data = {"losses": self.losses, "n_steps": self.n_steps}
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch=False, device=None):
        del end_of_epoch  # Unused in this class
        del device
        data = torch.load(path)
        self.losses = data["losses"]
        self.n_steps = data["n_steps"]


@checkpoints.register_checkpoint_hooks
class CyclicCosineScheduler:
    """The is an implementation of the Cyclic-Cosine learning rate scheduler with warmup.

    Reference:  https://openreview.net/pdf?id=BJYwwY9ll

    Note: this scheduler anneals the lr at each update of the model's weight,
    and n_steps must be saved for restarting.

    Arguments
    ---------
    lr_initial : float
        Initial learning rate (i.e. the lr used at epoch 0).
    n_warmup_steps : int
        Number of warm up steps.
    total_steps : int
        Total number of updating steps.

    Example
    -------
    >>> from speechbrain.nnet.linear import Linear
    >>> inp_tensor = torch.rand([1,660,3])
    >>> model = Linear(input_size=3, n_neurons=4)
    >>> optim = torch.optim.Adam(model.parameters(), lr=1)
    >>> output = model(inp_tensor)
    >>> scheduler =CyclicCosineScheduler(3, optim.param_groups[0]["lr"])
    >>> curr_lr,next_lr=scheduler(optim)
    >>> optim.param_groups[0]["lr"]
    0.9999999990130395
    >>> curr_lr,next_lr=scheduler(optim)
    >>> optim.param_groups[0]["lr"]
    0.9999999997532598
    >>> curr_lr,next_lr=scheduler(optim)
    >>> optim.param_groups[0]["lr"]
    1.0
    """

    def __init__(self, n_warmup_steps, lr_initial=None, total_steps=100000):
        self.n_warmup_steps = n_warmup_steps
        self.losses = []
        self.initial_lr = lr_initial
        self.current_lr = lr_initial
        self.total = total_steps

        self.n_steps = 0
        self.normalize = 1 / (n_warmup_steps * n_warmup_steps ** -1.5)

    def __call__(self, opt):
        """
        Arguments
        ---------
        opt : list of optimizers
            The optimizers to update using this scheduler.
        current_epoch : int
            Number of times the dataset has been iterated.
        current_loss : int
            A number for determining whether to change the learning rate.

        Returns
        -------
        current_lr : float
            The learning rate before the update.
        lr : float
            The learning rate after the update.
        """
        self.n_steps += 1

        if self.initial_lr is None:
            current_lr = opt.param_groups[0]["lr"]
        else:
            current_lr = self.current_lr

        lr = current_lr * self._get_lr_scale()

        # Changing the learning rate within the optimizer
        for param_group in opt.param_groups:
            param_group["lr"] = lr

        self.current_lr = current_lr
        return current_lr, lr

    def _get_lr_scale(self):
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return 0.5 * (
            math.cos(math.pi * (n_steps - n_warmup_steps) / self.total) + 1
        )

    @checkpoints.mark_as_saver
    def save(self, path):
        data = {"losses": self.losses, "n_steps": self.n_steps}
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch=False, device=None):
        del end_of_epoch  # Unused in this class
        del device  # Unused here
        data = torch.load(path)
        self.losses = data["losses"]
        self.n_steps = data["n_steps"]


@checkpoints.register_checkpoint_hooks
class ReduceLROnPlateau:
    """Learning rate scheduler which decreases the learning rate if the loss
    function of interest gets stuck on a plateau, or starts to increase.
    The difference from NewBobLRScheduler is that, this one keeps a memory of
    the last step where do not observe improvement, and compares against that
    particular loss value as opposed to the most recent loss.

    Arguments
    ---------
    lr_min : float
        The minimum allowable learning rate.
    factor : float
        Factor with which to reduce the learning rate.
    patience : int
        How many epochs to wait before reducing the learning rate.

    Example
    -------
    >>> from torch.optim import Adam
    >>> from speechbrain.nnet.linear import Linear
    >>> inp_tensor = torch.rand([1,660,3])
    >>> model = Linear(n_neurons=10, input_size=3)
    >>> optim = Adam(lr=1.0, params=model.parameters())
    >>> output = model(inp_tensor)
    >>> scheduler = ReduceLROnPlateau(0.25, 0.5, 2, 1)
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=1, current_loss=10.0)
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=2, current_loss=11.0)
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=3, current_loss=13.0)
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=4, current_loss=14.0)
    >>> next_lr
    0.5
    """

    def __init__(
        self, lr_min=1e-8, factor=0.5, patience=2, dont_halve_until_epoch=65
    ):
        self.lr_min = lr_min
        self.factor = factor
        self.patience = patience
        self.patience_counter = 0
        self.losses = []
        self.dont_halve_until_epoch = dont_halve_until_epoch
        self.anchor = 99999

    def __call__(self, optim_list, current_epoch, current_loss):
        """
        Arguments
        ---------
        optim_list : list of optimizers
            The optimizers to update using this scheduler.
        current_epoch : int
            Number of times the dataset has been iterated.
        current_loss : int
            A number for determining whether to change the learning rate.

        Returns
        -------
        current_lr : float
            The learning rate before the update.
        next_lr : float
            The learning rate after the update.
        """
        for opt in optim_list:
            current_lr = opt.param_groups[0]["lr"]

            if current_epoch <= self.dont_halve_until_epoch:
                next_lr = current_lr
                self.anchor = current_loss
            else:
                if current_loss <= self.anchor:
                    self.patience_counter = 0
                    next_lr = current_lr
                    self.anchor = current_loss
                elif (
                    current_loss > self.anchor
                    and self.patience_counter < self.patience
                ):
                    self.patience_counter = self.patience_counter + 1
                    next_lr = current_lr
                else:
                    next_lr = current_lr * self.factor
                    self.patience_counter = 0

            # impose the lower bound
            next_lr = max(next_lr, self.lr_min)

        # Updating current loss
        self.losses.append(current_loss)

        return current_lr, next_lr

    @checkpoints.mark_as_saver
    def save(self, path):
        data = {
            "losses": self.losses,
            "anchor": self.anchor,
            "patience_counter": self.patience_counter,
        }
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch=False, device=None):
        del end_of_epoch  # Unused in this class
        del device  # Not used
        data = torch.load(path)
        self.losses = data["losses"]
        self.anchor = data["anchor"]
        self.patience_counter = data["patience_counter"]


@checkpoints.register_checkpoint_hooks
class CyclicLRScheduler:
    """This implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.

    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see the reference paper.

    Arguments
    ---------
    base_lr : float
        initial learning rate which is the
        lower boundary in the cycle.
    max_lr : float
        upper boundary in the cycle. Functionally,
        it defines the cycle amplitude (max_lr - base_lr).
        The lr at any cycle is the sum of base_lr
        and some scaling of the amplitude; therefore
        max_lr may not actually be reached depending on
        scaling function.
    step_size : int
        number of training iterations per
        half cycle. The authors suggest setting step_size
        2-8 x training iterations in epoch.
    mode : str
        one of {triangular, triangular2, exp_range}.
        Default 'triangular'.
        Values correspond to policies detailed above.
        If scale_fn is not None, this argument is ignored.
    gamma : float
        constant in 'exp_range' scaling function:
        gamma**(cycle iterations)
    scale_fn : lambda function
        Custom scaling policy defined by a single
        argument lambda function, where
        0 <= scale_fn(x) <= 1 for all x >= 0.
        mode parameter is ignored
    scale_mode : str
        {'cycle', 'iterations'}.
        Defines whether scale_fn is evaluated on
        cycle number or cycle iterations (training
        iterations since start of cycle). Default is 'cycle'.

    Example
    -------
    >>> from speechbrain.nnet.linear import Linear
    >>> inp_tensor = torch.rand([1,660,3])
    >>> model = Linear(input_size=3, n_neurons=4)
    >>> optim = torch.optim.Adam(model.parameters(), lr=1)
    >>> output = model(inp_tensor)
    >>> scheduler = CyclicLRScheduler(base_lr=0.1, max_lr=0.3, step_size=2)
    >>> scheduler.on_batch_end(optim)
    >>> optim.param_groups[0]["lr"]
    0.2
    >>> scheduler.on_batch_end(optim)
    >>> optim.param_groups[0]["lr"]
    0.3
    >>> scheduler.on_batch_end(optim)
    >>> optim.param_groups[0]["lr"]
    0.2
    """

    def __init__(
        self,
        base_lr=0.001,
        max_lr=0.006,
        step_size=2000.0,
        mode="triangular",
        gamma=1.0,
        scale_fn=None,
        scale_mode="cycle",
    ):
        super(CyclicLRScheduler, self).__init__()

        self.losses = []
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == "triangular":
                self.scale_fn = lambda x: 1.0
                self.scale_mode = "cycle"
            elif self.mode == "triangular2":
                self.scale_fn = lambda x: 1 / (2.0 ** (x - 1))
                self.scale_mode = "cycle"
            elif self.mode == "exp_range":
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = "iterations"
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.0

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None, new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.0

    def __call__(self, epoch):
        old_lr = self.current_lr
        new_lr = self.clr(self.clr_iterations + 1)

        return old_lr, new_lr

    def clr(self, clr_iterations):
        cycle = math.floor(1 + clr_iterations / (2 * self.step_size))
        x = abs(clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == "cycle":
            return self.base_lr + (self.max_lr - self.base_lr) * max(
                0, (1 - x)
            ) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * max(
                0, (1 - x)
            ) * self.scale_fn(clr_iterations)

    def on_batch_end(self, opt):
        """
        Arguments
        ---------
        opt : optimizers
            The optimizers to update using this scheduler.
        """
        self.clr_iterations += 1

        lr = self.clr(self.clr_iterations)
        # current_lr = opt.param_groups[0]["lr"]
        current_lr = opt.get_lr()

        # Changing the learning rate within the optimizer
        # todo 暂时不设置学习率，而是根据学习的过程中自动调整
        # for param_group in opt.param_groups:
            # param_group["lr"] = lr
        opt.set_lr(lr)
        self.current_lr = current_lr

    @checkpoints.mark_as_saver
    def save(self, path):
        data = {"losses": self.losses, "clr_iterations": self.clr_iterations}
        paddle.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch=False, device=None):
        del end_of_epoch  # Unused in this class
        del device
        data = paddle.load(path)
        self.losses = data["losses"]
        self.clr_iterations = data["clr_iterations"]
