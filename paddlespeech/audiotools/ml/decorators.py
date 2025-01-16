# MIT License, Copyright (c) 2023-Present, Descript.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Modified from audiotools(https://github.com/descriptinc/audiotools/blob/master/audiotools/ml/decorators.py)
import math
import os
import time
from collections import defaultdict
from functools import wraps

import paddle
import paddle.distributed as dist
from rich import box
from rich.console import Console
from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import BarColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn
from rich.rule import Rule
from rich.table import Table
from visualdl import LogWriter


# This is here so that the history can be pickled.
def default_list():
    return []


class Mean:
    """Keeps track of the running mean, along with the latest
    value.
    """

    def __init__(self):
        self.reset()

    def __call__(self):
        mean = self.total / max(self.count, 1)
        return mean

    def reset(self):
        self.count = 0
        self.total = 0

    def update(self, val):
        if math.isfinite(val):
            self.count += 1
            self.total += val


def when(condition):
    """Runs a function only when the condition is met. The condition is
    a function that is run.

    Parameters
    ----------
    condition : Callable
        Function to run to check whether or not to run the decorated
        function.

    Example
    -------
    Checkpoint only runs every 100 iterations, and only if the
    local rank is 0.

    >>> i = 0
    >>> rank = 0
    >>>
    >>> @when(lambda: i % 100 == 0 and rank == 0)
    >>> def checkpoint():
    >>>     print("Saving to /runs/exp1")
    >>>
    >>> for i in range(1000):
    >>>     checkpoint()

    """

    def decorator(fn):
        @wraps(fn)
        def decorated(*args, **kwargs):
            if condition():
                return fn(*args, **kwargs)

        return decorated

    return decorator


def timer(prefix: str="time"):
    """Adds execution time to the output dictionary of the decorated
    function. The function decorated by this must output a dictionary.
    The key added will follow the form "[prefix]/[name_of_function]"

    Parameters
    ----------
    prefix : str, optional
        The key added will follow the form "[prefix]/[name_of_function]",
        by default "time".
    """

    def decorator(fn):
        @wraps(fn)
        def decorated(*args, **kwargs):
            s = time.perf_counter()
            output = fn(*args, **kwargs)
            assert isinstance(output, dict)
            e = time.perf_counter()
            output[f"{prefix}/{fn.__name__}"] = e - s
            return output

        return decorated

    return decorator


class Tracker:
    """
    A tracker class that helps to monitor the progress of training and logging the metrics.

    Attributes
    ----------
    metrics : dict
        A dictionary containing the metrics for each label.
    history : dict
        A dictionary containing the history of metrics for each label.
    writer : LogWriter
        A LogWriter object for logging the metrics.
    rank : int
        The rank of the current process.
    step : int
        The current step of the training.
    tasks : dict
        A dictionary containing the progress bars and tables for each label.
    pbar : Progress
        A progress bar object for displaying the progress.
    consoles : list
        A list of console objects for logging.
    live : Live
        A Live object for updating the display live.

    Methods
    -------
    print(msg: str)
        Prints the given message to all consoles.
    update(label: str, fn_name: str)
        Updates the progress bar and table for the given label.
    done(label: str, title: str)
        Resets the progress bar and table for the given label and prints the final result.
    track(label: str, length: int, completed: int = 0, op: dist.ReduceOp = dist.ReduceOp.AVG, ddp_active: bool = "LOCAL_RANK" in os.environ)
        A decorator for tracking the progress and metrics of a function.
    log(label: str, value_type: str = "value", history: bool = True)
        A decorator for logging the metrics of a function.
    is_best(label: str, key: str) -> bool
        Checks if the latest value of the given key in the label is the best so far.
    state_dict() -> dict
        Returns a dictionary containing the state of the tracker.
    load_state_dict(state_dict: dict) -> Tracker
        Loads the state of the tracker from the given state dictionary.
    """

    def __init__(
            self,
            writer: LogWriter=None,
            log_file: str=None,
            rank: int=0,
            console_width: int=100,
            step: int=0, ):
        """
        Initializes the Tracker object.

        Parameters
        ----------
        writer : LogWriter, optional
            A LogWriter object for logging the metrics, by default None.
        log_file : str, optional
            The path to the log file, by default None.
        rank : int, optional
            The rank of the current process, by default 0.
        console_width : int, optional
            The width of the console, by default 100.
        step : int, optional
            The current step of the training, by default 0.
        """
        self.metrics = {}
        self.history = {}
        self.writer = writer
        self.rank = rank
        self.step = step

        # Create progress bars etc.
        self.tasks = {}
        self.pbar = Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            "{task.completed}/{task.total}",
            BarColumn(),
            TimeElapsedColumn(),
            "/",
            TimeRemainingColumn(), )
        self.consoles = [Console(width=console_width)]
        self.live = Live(console=self.consoles[0], refresh_per_second=10)
        if log_file is not None:
            self.consoles.append(
                Console(width=console_width, file=open(log_file, "a")))

    def print(self, msg):
        """
        Prints the given message to all consoles.

        Parameters
        ----------
        msg : str
            The message to be printed.
        """
        if self.rank == 0:
            for c in self.consoles:
                c.log(msg)

    def update(self, label, fn_name):
        """
        Updates the progress bar and table for the given label.

        Parameters
        ----------
        label : str
            The label of the progress bar and table to be updated.
        fn_name : str
            The name of the function associated with the label.
        """
        if self.rank == 0:
            self.pbar.advance(self.tasks[label]["pbar"])

            # Create table
            table = Table(title=label, expand=True, box=box.MINIMAL)
            table.add_column("key", style="cyan")
            table.add_column("value", style="bright_blue")
            table.add_column("mean", style="bright_green")

            keys = self.metrics[label]["value"].keys()
            for k in keys:
                value = self.metrics[label]["value"][k]
                mean = self.metrics[label]["mean"][k]()
                table.add_row(k, f"{value:10.6f}", f"{mean:10.6f}")

            self.tasks[label]["table"] = table
            tables = [t["table"] for t in self.tasks.values()]
            group = Group(*tables, self.pbar)
            self.live.update(
                Group(
                    Padding("", (0, 0)),
                    Rule(f"[italic]{fn_name}()", style="white"),
                    Padding("", (0, 0)),
                    Panel.fit(
                        group,
                        padding=(0, 5),
                        title="[b]Progress",
                        border_style="blue", ), ))

    def done(self, label: str, title: str):
        """
        Resets the progress bar and table for the given label and prints the final result.

        Parameters
        ----------
        label : str
            The label of the progress bar and table to be reset.
        title : str
            The title to be displayed when printing the final result.
        """
        for label in self.metrics:
            for v in self.metrics[label]["mean"].values():
                v.reset()

        if self.rank == 0:
            self.pbar.reset(self.tasks[label]["pbar"])
            tables = [t["table"] for t in self.tasks.values()]
            group = Group(Markdown(f"# {title}"), *tables, self.pbar)
            self.print(group)

    def track(
            self,
            label: str,
            length: int,
            completed: int=0,
            op: dist.ReduceOp=dist.ReduceOp.AVG,
            ddp_active: bool="LOCAL_RANK" in os.environ, ):
        """
        A decorator for tracking the progress and metrics of a function.

        Parameters
        ----------
        label : str
            The label to be associated with the progress and metrics.
        length : int
            The total number of iterations to be completed.
        completed : int, optional
            The number of iterations already completed, by default 0.
        op : dist.ReduceOp, optional
            The reduce operation to be used, by default dist.ReduceOp.AVG.
        ddp_active : bool, optional
            Whether the DistributedDataParallel is active, by default "LOCAL_RANK" in os.environ.
        """
        self.tasks[label] = {
            "pbar":
            self.pbar.add_task(
                f"[white]Iteration ({label})",
                total=length,
                completed=completed),
            "table":
            Table(),
        }
        self.metrics[label] = {
            "value": defaultdict(),
            "mean": defaultdict(lambda: Mean()),
        }

        def decorator(fn):
            @wraps(fn)
            def decorated(*args, **kwargs):
                output = fn(*args, **kwargs)
                if not isinstance(output, dict):
                    self.update(label, fn.__name__)
                    return output
                # Collect across all DDP processes
                scalar_keys = []
                for k, v in output.items():
                    if isinstance(v, (int, float)):
                        v = paddle.to_tensor([v])
                    if not paddle.is_tensor(v):
                        continue
                    if ddp_active and v.is_cuda:
                        dist.all_reduce(v, op=op)
                    output[k] = v.detach()
                    if paddle.numel(v) == 1:
                        scalar_keys.append(k)
                        output[k] = v.item()

                # Save the outputs to tracker
                for k, v in output.items():
                    if k not in scalar_keys:
                        continue
                    self.metrics[label]["value"][k] = v
                    # Update the running mean
                    self.metrics[label]["mean"][k].update(v)

                self.update(label, fn.__name__)
                return output

            return decorated

        return decorator

    def log(self, label: str, value_type: str="value", history: bool=True):
        """
        A decorator for logging the metrics of a function.

        Parameters
        ----------
        label : str
            The label to be associated with the logging.
        value_type : str, optional
            The type of value to be logged, by default "value".
        history : bool, optional
            Whether to save the history of the metrics, by default True.
        """
        assert value_type in ["mean", "value"]
        if history:
            if label not in self.history:
                self.history[label] = defaultdict(default_list)

        def decorator(fn):
            @wraps(fn)
            def decorated(*args, **kwargs):
                output = fn(*args, **kwargs)
                if self.rank == 0:
                    nonlocal value_type, label
                    metrics = self.metrics[label][value_type]
                    for k, v in metrics.items():
                        v = v() if isinstance(v, Mean) else v
                        if self.writer is not None:
                            self.writer.add_scalar(
                                tag=f"{k}/{label}", value=v, step=self.step)
                        if label in self.history:
                            self.history[label][k].append(v)

                    if label in self.history:
                        self.history[label]["step"].append(self.step)

                return output

            return decorated

        return decorator

    def is_best(self, label, key):
        """
        Checks if the latest value of the given key in the label is the best so far.

        Parameters
        ----------
        label : str
            The label of the metrics to be checked.
        key : str
            The key of the metric to be checked.

        Returns
        -------
        bool
            True if the latest value is the best so far, otherwise False.
        """
        return self.history[label][key][-1] == min(self.history[label][key])

    def state_dict(self):
        """
        Returns a dictionary containing the state of the tracker.

        Returns
        -------
        dict
            A dictionary containing the history and step of the tracker.
        """
        return {"history": self.history, "step": self.step}

    def load_state_dict(self, state_dict):
        """
        Loads the state of the tracker from the given state dictionary.

        Parameters
        ----------
        state_dict : dict
            A dictionary containing the history and step of the tracker.

        Returns
        -------
        Tracker
            The tracker object with the loaded state.
        """
        self.history = state_dict["history"]
        self.step = state_dict["step"]
        return self
