"""Utilities for hyperparameter optimization.
This wrapper has an optional dependency on
Oríon

https://orion.readthedocs.io/en/stable/
https://github.com/Epistimio/orion

Authors
 * Artem Ploujnikov 2021
"""
import importlib
import logging
import json
import os
import speechbrain as sb
import sys

from datetime import datetime
from hyperpyyaml import load_hyperpyyaml


logger = logging.getLogger(__name__)

MODULE_ORION = "orion.client"
FORMAT_TIMESTAMP = "%Y%m%d%H%M%S%f"
DEFAULT_TRIAL_ID = "hpopt"
DEFAULT_REPORTER = "generic"
ORION_TRIAL_ID_ENV = [
    "ORION_EXPERIMENT_NAME",
    "ORION_EXPERIMENT_VERSION",
    "ORION_TRIAL_ID",
]
KEY_HPOPT = "hpopt"
KEY_HPOPT_MODE = "hpopt_mode"

_hpopt_modes = {}


def hpopt_mode(mode):
    """A decorator to register a reporter implementation for
    a hyperparameter optimization mode

    Arguments
    ---------
    mode: str
        the mode to register

    Returns
    -------
    f: callable
        a callable function that registers and returns the
        reporter class

    Example
    -------
    >>> @hpopt_mode("raw")
    ... class RawHyperparameterOptimizationReporter(HyperparameterOptimizationReporter):
    ...    def __init__(self, *args, **kwargs):
    ...        super().__init__(    *args, **kwargs)
    ...    def report_objective(self, result):
    ...        objective = result[self.objective_key]
    ...        print(f"Objective: {objective}")

    >>> reporter = get_reporter("raw", objective_key="error")
    >>> result = {"error": 1.2, "train_loss": 7.2}
    >>> reporter.report_objective(result)
    Objective: 1.2
    """

    def f(cls):
        _hpopt_modes[mode] = cls
        return cls

    return f


class HyperparameterOptimizationReporter:
    """A base class for hyperparameter fit reporters

    Arguments
    ---------
    objective_key: str
        the key from the result dictionary to be used as the objective
    """

    def __init__(self, objective_key):
        self.objective_key = objective_key

    def report_objective(self, result):
        """Reports the objective for hyperparameter optimization.

        Arguments
        ---------
        result: dict
            a dictionary with the run result.
        """
        return NotImplemented

    @property
    def is_available(self):
        """Determines whether this reporter is available"""
        return True

    @property
    def trial_id(self):
        """The unique ID of this trial (used for folder naming)"""
        return DEFAULT_TRIAL_ID


@hpopt_mode("generic")
class GenericHyperparameterOptimizationReporter(
    HyperparameterOptimizationReporter
):
    """
    A generic hyperparameter fit reporter that outputs the result as
    JSON to an arbitrary data stream, which may be read as a third-party
    tool

    Arguments
    ---------
    objective_key: str
        the key from the result dictionary to be used as the objective

    """

    def __init__(self, reference_date=None, output=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output = output or sys.stdout
        self.reference_date = reference_date
        self._trial_id = None

    def report_objective(self, result):
        """Reports the objective for hyperparameter optimization.

        Arguments
        ---------
        result: dict
            a dictionary with the run result.

        Example
        -------
        >>> reporter = GenericHyperparameterOptimizationReporter(
        ...     objective_key="error"
        ... )
        >>> result = {"error": 1.2, "train_loss": 7.2}
        >>> reporter.report_objective(result)
        {"error": 1.2, "train_loss": 7.2, "objective": 1.2}
        """
        json.dump(
            dict(result, objective=result[self.objective_key]), self.output
        )

    @property
    def trial_id(self):
        """The unique ID of this trial (used mainly for folder naming)

        Example
        -------
        >>> import datetime
        >>> reporter = GenericHyperparameterOptimizationReporter(
        ...     objective_key="error",
        ...     reference_date=datetime.datetime(2021, 1, 3)
        ... )
        >>> print(reporter.trial_id)
        20210103000000000000
        """
        if self._trial_id is None:
            date = self.reference_date or datetime.now()
            self._trial_id = date.strftime(FORMAT_TIMESTAMP)
        return self._trial_id


@hpopt_mode("orion")
class OrionHyperparameterOptimizationReporter(
    HyperparameterOptimizationReporter
):
    """A result reporter implementation based on Orion

    Arguments
    ---------
    orion_client: module
        the Python module for Orion
    """

    def __init__(self, objective_key):
        super().__init__(objective_key=objective_key)
        self.orion_client = None
        self._trial_id = None
        self._check_client()

    def _check_client(self):
        try:
            self.orion_client = importlib.import_module(MODULE_ORION)
        except ImportError:
            logger.warning("Orion is not available")
            self.orion_client = None

    def _format_message(self, result):
        """Formats the log message for output

        Arguments
        ---------
        result: dict
            the result dictionary

        Returns
        -------
        message: str
            a formatted message"""
        return ", ".join(f"{key} = {value}" for key, value in result.items())

    def report_objective(self, result):
        """Reports the objective for hyperparameter optimization.

        Arguments
        ---------
        result: dict
            a dictionary with the run result.
        """
        message = self._format_message(result)
        logger.info(f"Hyperparameter fit: {message}")
        if self.orion_client is not None:
            objective_value = result[self.objective_key]
            self.orion_client.report_objective(objective_value)

    @property
    def trial_id(self):
        """The unique ID of this trial (used mainly for folder naming)"""
        if self._trial_id is None:
            self._trial_id = "-".join(
                os.getenv(name) or "" for name in ORION_TRIAL_ID_ENV
            )
        return self._trial_id

    @property
    def is_available(self):
        """Determines if Orion is available. In order for it to
        be available, the library needs to be installed, and at
        least one of ORION_EXPERIMENT_NAME, ORION_EXPERIMENT_VERSION,
        ORION_TRIAL_ID needs to be set"""
        return self.orion_client is not None and any(
            os.getenv(name) for name in ORION_TRIAL_ID_ENV
        )


def get_reporter(mode, *args, **kwargs):
    """Attempts to get the reporter specified by the mode
    and reverts to a generic one if it is not available

    Arguments
    ---------
    mode: str
        a string identifier for a registered hyperparametr
        optimization mode, corresponding to a specific reporter
        instance

    Returns
    -------
    reporter: HyperparameterOptimizationReporter
        a reporter instance

    Example
    -------
    >>> reporter = get_reporter("generic", objective_key="error")
    >>> result = {"error": 3.4, "train_loss": 1.2}
    >>> reporter.report_objective(result)
    {"error": 3.4, "train_loss": 1.2, "objective": 3.4}
    """
    reporter_cls = _hpopt_modes.get(mode)
    if reporter_cls is None:
        logger.warn(f"hpopt_mode {mode} is not supported, reverting to generic")
        reporter_cls = _hpopt_modes[DEFAULT_REPORTER]
    reporter = reporter_cls(*args, **kwargs)
    if not reporter.is_available:
        logger.warn("Reverting to a generic reporter")
        reporter_cls = _hpopt_modes[DEFAULT_REPORTER]
        reporter = reporter_cls(*args, **kwargs)
    return reporter


_context = {"current": None}


class HyperparameterOptimizationContext:
    """
    A convenience context manager that makes it possible to conditionally
    enable hyperparameter optimization for a recipe.

    Arguments
    ---------
    reporter_args: list
        arguments to the reporter class
    reporter_kwargs: dict
        keyword arguments to the reporter class

    Example
    -------
    >>> ctx = HyperparameterOptimizationContext(
    ...     reporter_args=[],
    ...     reporter_kwargs={"objective_key": "error"}
    ... )
    """

    def __init__(self, reporter_args=None, reporter_kwargs=None):
        self.reporter_args = reporter_args or []
        self.reporter_kwargs = reporter_kwargs or {}
        self.reporter = None
        self.enabled = False
        self.result = {"objective": 0.0}

    def parse_arguments(self, arg_list):
        """A version of speechbrain.parse_arguments enhanced for hyperparameter optimization.

        If a parameter named 'hpopt' is provided, hyperparameter
        optimization and reporting will be enabled.

        If the parameter value corresponds to a filename, it will
        be read as a hyperpyaml file, and the contents will be added
        to "overrides". This is useful for cases where the values of
        certain hyperparameters are different during hyperparameter
        optimization vs during full training (e.g. number of epochs, saving
        files, etc)

        Arguments
        ---------
        arg_list: a list of arguments

        Returns
        -------
        param_file : str
            The location of the parameters file.
        run_opts : dict
            Run options, such as distributed, device, etc.
        overrides : dict
            The overrides to pass to ``load_hyperpyyaml``.

        Example
        -------
        >>> ctx = HyperparameterOptimizationContext()
        >>> arg_list = ["hparams.yaml", "--x", "1", "--y", "2"]
        >>> hparams_file, run_opts, overrides = ctx.parse_arguments(arg_list)
        >>> print(f"File: {hparams_file}, Overrides: {overrides}")
        File: hparams.yaml, Overrides: {'x': 1, 'y': 2}
        """
        hparams_file, run_opts, overrides_yaml = sb.parse_arguments(arg_list)
        overrides = load_hyperpyyaml(overrides_yaml)
        hpopt = overrides.get(KEY_HPOPT, False)
        hpopt_mode = overrides.get(KEY_HPOPT_MODE) or DEFAULT_REPORTER
        if hpopt:
            self.enabled = True
            self.reporter = get_reporter(
                hpopt_mode, *self.reporter_args, **self.reporter_kwargs
            )
            if isinstance(hpopt, str) and os.path.exists(hpopt):
                with open(hpopt) as hpopt_file:
                    trial_id = get_trial_id()
                    hpopt_overrides = load_hyperpyyaml(
                        hpopt_file,
                        overrides={"trial_id": trial_id},
                        overrides_must_match=False,
                    )
                    overrides = dict(hpopt_overrides, **overrides)
                    for key in [KEY_HPOPT, KEY_HPOPT_MODE]:
                        if key in overrides:
                            del overrides[key]
        return hparams_file, run_opts, overrides

    def __enter__(self):
        _context["current"] = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None and self.result is not None:
            reporter = self.reporter
            if not reporter:
                reporter = get_reporter(
                    hpopt_mode, *self.reporter_args, **self.reporter_kwargs
                )
            reporter.report_objective(self.result)
        _context["current"] = None


def hyperparameter_optimization(*args, **kwargs):
    """Initializes the hyperparameter optimization context

    Example
    -------
    >>> import sys
    >>> with hyperparameter_optimization(objective_key="error", output=sys.stdout) as hp_ctx:
    ...     result = {"error": 3.5, "train_loss": 2.1}
    ...     report_result(result)
    ...
    {"error": 3.5, "train_loss": 2.1, "objective": 3.5}
    """
    hpfit = HyperparameterOptimizationContext(args, kwargs)
    return hpfit


def report_result(result):
    """Reports the result using the current reporter, if available.
    When not in hyperparameter optimization mode, this function does nothing.

    Arguments
    ---------
    result: dict
        A dictionary of stats to be reported

    Example
    -------
    >>> result = {"error": 3.5, "train_loss": 2.1}
    >>> report_result(result["error"])
    """
    ctx = _context["current"]
    if ctx:
        ctx.result = result


def get_trial_id():
    """
    Returns the ID of the current hyperparameter optimization trial,
    used primarily for the name of experiment folders.

    When using a context, the convention for identifying the trial ID
    will depend on the reporter being used. The default implementation
    returns a fixed value ("hpopt")

    Returns
    -------
    trial_id: str
        the trial identifier

    Example
    -------
    >>> trial_id = get_trial_id()
    >>> trial_id
    'hpopt'
    """
    ctx = _context["current"]
    trial_id = ctx.reporter.trial_id if ctx else DEFAULT_TRIAL_ID
    return trial_id
