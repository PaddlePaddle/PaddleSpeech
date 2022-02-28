"""This module implements a checkpoint saver and loader.

A checkpoint in an experiment usually needs to save the state of many different
things: the model parameters, optimizer parameters, what epoch is this, etc.
The save format for a checkpoint is a directory, where each of these separate
saveable things gets its own file. Additionally, a special file holds meta
information about the checkpoint (by default just time of creation, but you
can specify anything else you may wish, e.g. validation loss).

The interface for the checkpoint system requires you to specify what things to
save. This approach is flexible and agnostic of how your experiment is actually
run.

The interface requires you to specify names for each thing to save. This name
is used to give the right parameter file to the right object when recovering.

Default saving and loading methods are only added for torch.nn.Modules (and
their subclasses), and torch.optim.Optimizers. If those methods do not work for
your object, you can specify your own saving and/or loading methods, either for
a particular instance or a for a class.

Example
-------
>>> # Toy example Module:
>>> class Recoverable(paddle.nn.Layer):
...     def __init__(self, param):
...         super().__init__()
...         self.param = torch.nn.Parameter(torch.tensor([param]))
...     def forward(self, x):
...         return x * self.param
>>> model = Recoverable(1.)
>>> tempdir = getfixture('tmpdir')
>>> # In simple cases, the module aims to have a terse syntax,
>>> # consisting of three steps.
>>> # 1. Specifying where to save checkpoints and what is included in a
>>> # checkpoint:
>>> checkpointer = Checkpointer(tempdir, {"network": model})
>>> # 2. Recover from the latest checkpoint, if one is found:
>>> checkpointer.recover_if_possible()
>>> # Run your experiment:
>>> data = [(0.1, 0.9), (0.3, 0.8)]
>>> for example, target in data:
...     loss = (model(example) - target)**2
...     # 3. Save checkpoints, and keep by default just one, the newest:
...     ckpt = checkpointer.save_and_keep_only()

Authors
 * Aku Rouhe 2020
"""
import paddle
import collections
import collections.abc
import os
import time
import yaml
import pathlib
import inspect
import shutil
import logging
import warnings

logger = logging.getLogger(__name__)

CKPT_PREFIX = "PDPARAMS"
METAFNAME = f"{CKPT_PREFIX}.yaml"  # Important that this is not .ckpt
PARAMFILE_EXT = ".pdparams"  # ...because these files will be


def paddle_recovery(obj, path, end_of_epoch, device=None):
    """Loads a paddle.nn.Layer state_dict from the given path instantly.

    This can be made the default for torch.nn.Modules with:
    >>> DEFAULT_LOAD_HOOKS[paddle.nn.Layer] = torch_recovery

    Arguments
    ---------
    obj : paddle.nn.Layer
        Instance for which to load the parameters.
    path : str, pathlib.Path
        Path where to load from.
    end_of_epoch : bool
        Whether the recovery comes from an end of epoch checkpoint.
    device : str
        Torch device, where to map the loaded parameters.

    Returns
    -------
    None
        Given object is modified in place.
    """
    del end_of_epoch  # Unused
    print("paddle load")
    try:
        # obj.load_state_dict(paddle.load(path, map_location=device), strict=True)
        # 
        path = str(path)
        # print("path type: {}".format(type(path)))
        layer_state_dict = paddle.load(str(path))
        # print("obj state dict: {}".format(obj.state_dict()))
        obj.set_state_dict(layer_state_dict)
        
    except TypeError:
        # print("load data: {}".format(paddle.load(str(path))))
        obj.set_state_dict(paddle.load(path))


def paddle_save(obj, path):
    """Saves the obj's parameters to path.

    Default save hook for torch.nn.Modules
    For saving paddle.nn.Layer state_dicts.

    Arguments
    ---------
    obj : paddle.nn.Layer
        Instance to save.
    path : str, pathlib.Path
        Path where to save to.

    Returns
    -------
    None
        State dict is written to disk.
    """
    state_dict = obj.state_dict()
    if not state_dict:
        logger.warning(f"Saving an empty state_dict for {obj} in {path}.")
    paddle.save(state_dict, path)


def paddle_parameter_transfer(obj, path, device):
    """Non-strict Torch Module state_dict load.

    Loads a set of parameters from path to obj. If obj has layers for which
    parameters can't be found, only a warning is logged. Same thing
    if the path has parameters for layers which don't find a counterpart
    in obj.

    Arguments
    ---------
    obj : paddle.nn.Layer
        Instance for which to load the parameters.
    path : str
        Path where to load from.

    Returns
    -------
    None
        The object is modified in place.
    """
    # path = str(path)
    # load_state_dict = paddle.load(path)
    incompatible_keys = obj.set_state_dict(
        paddle.load(str(path))
    )
    # todo 目前在Paddle的set_state_dict中，并没有返回值
    # 如果模型不匹配，可能会直接报错，并不知道哪里有问题
    # for missing_key in incompatible_keys.missing_keys:
    #     logger.warning(
    #         f"During parameter transfer to {obj} loading from "
    #         + f"{path}, the transferred parameters did not have "
    #         + f"parameters for the key: {missing_key}"
    #     )
    # for unexpected_key in incompatible_keys.unexpected_keys:
    #     logger.warning(
    #         f"During parameter transfer to {obj} loading from "
    #         + f"{path}, the object could not use the parameters loaded "
    #         + f"with the key: {unexpected_key}"
    #     )


# These dicts are indexed by class and hold the default checkpoints methods
DEFAULT_LOAD_HOOKS = {
    paddle.nn.Layer: paddle_recovery,
    paddle.optimizer.Optimizer: paddle_recovery,
    # paddle.optimizer.lr._LRScheduler: paddle_recovery,
    # paddle.optimizer.lr.ReduceLROnPlateau: paddle_recovery,
}
DEFAULT_SAVE_HOOKS = {
    paddle.nn.Layer: paddle_save,
    paddle.optimizer.Optimizer: paddle_save,
    # paddle.optimizer.lr._LRScheduler: paddle_save,
    # paddle.optimizer.lr.ReduceLROnPlateau: paddle_save,
}
DEFAULT_TRANSFER_HOOKS = {
    paddle.nn.Layer: paddle_parameter_transfer,
}

# Add a transfer hook for sentencepiece if it is installed:
try:
    import sentencepiece as spm

    def _load_spm(obj, path, device=None):
        obj.load(str(path))  # SentencePieceProcessor needs a string.

    DEFAULT_TRANSFER_HOOKS[spm.SentencePieceProcessor] = _load_spm
    del spm  # Don't leave it here bare.
except ImportError:
    # SentencePiece not loaded, fine!
    pass


def mark_as_saver(method):
    """Method decorator which marks given method as the checkpoint saving hook.

    See register_checkpoint_hooks for example.

    Arguments
    ---------
    method : callable
        Method of the class to decorate. Must be callable with
        signature (instance, path) using positional arguments. This is
        satisfied by for example: def saver(self, path):

    Note
    ----
    This will not add the hook (not possible via a method decorator),
    you must also decorate the class with @register_checkpoint_hooks
    Only one method can be added as the hook.
    """
    sig = inspect.signature(method)
    try:
        sig.bind(object(), pathlib.Path("testpath"))
    except TypeError:
        MSG = "Checkpoint saver must match signature (instance, path)"
        raise TypeError(MSG)
    method._speechbrain_saver = True
    return method


def mark_as_loader(method):
    """Method decorator which marks given method as checkpoint loading hook.

    Arguments
    ---------
    method : callable
        Method of the class to decorate. Must be callable with
        signature (instance, path, end_of_epoch, device) using positional
        arguments. This is satisfied by for example:
        `def loader(self, path, end_of_epoch, device):`

    Note
    ----
    This will not add the hook (not possible via a method decorator),
    you must also decorate the class with @register_checkpoint_hooks
    Only one method can be added as the hook.
    """
    sig = inspect.signature(method)
    try:
        sig.bind(object(), pathlib.Path("testpath"), True, None)
    except TypeError:
        MSG = "Checkpoint loader must have signature (self, path, end_of_epoch, device)"
        raise TypeError(MSG)
    method._speechbrain_loader = True
    return method


def mark_as_transfer(method):
    """Method decorator which marks given method as a parameter transfer hook.

    Arguments
    ---------
    method : callable
        Method of the class to decorate. Must be callable with
        signature (instance, path, device) using positional
        arguments. This is satisfied by for example:
        `def loader(self, path, device):`

    Note
    ----
    This will not add the hook (not possible via a method decorator),
    you must also decorate the class with @register_checkpoint_hooks
    Only one method can be added as the hook.

    Note
    ----
    The transfer hook is prioritized over the loader hook by the ``Pretrainer``
    However, if no transfer hook is registered, the Pretrainer will use the
    loader hook.
    """
    sig = inspect.signature(method)
    try:
        sig.bind(object(), pathlib.Path("testpath"), device=None)
    except TypeError:
        MSG = "Transfer hook must have signature (self, path, device)"
        raise TypeError(MSG)
    method._speechbrain_transfer = True
    return method


def register_checkpoint_hooks(cls):
    """Class decorator which registers the load, save and transfer hooks.

    The hooks must have been marked with mark_as_loader and mark_as_saver,
    and possibly mark_as_transfer.

    Arguments
    ---------
    cls : class
        Class to decorate

    Example
    -------
    >>> @register_checkpoint_hooks
    ... class CustomRecoverable:
    ...     def __init__(self, param):
    ...         self.param = int(param)
    ...
    ...     @mark_as_saver
    ...     def save(self, path):
    ...         with open(path, "w") as fo:
    ...             fo.write(str(self.param))
    ...
    ...     @mark_as_loader
    ...     def load(self, path, end_of_epoch, device=None):
    ...         del end_of_epoch  # Unused here
    ...         with open(path) as fi:
    ...             self.param = int(fi.read())
    """
    global DEFAULT_LOAD_HOOKS
    global DEFAULT_SAVE_HOOKS
    global DEFAULT_TRANSFER_HOOKS
    for name, method in cls.__dict__.items():
        if hasattr(method, "_speechbrain_saver"):
            DEFAULT_SAVE_HOOKS[cls] = method
            logger.debug(f"Registered checkpoint save hook for {name}")
        if hasattr(method, "_speechbrain_loader"):
            DEFAULT_LOAD_HOOKS[cls] = method
            logger.debug(f"Registered checkpoint load hook for {name}")
        if hasattr(method, "_speechbrain_transfer"):
            DEFAULT_TRANSFER_HOOKS[cls] = method
            logger.debug(f"Registered parameter transfer hook for {name}")
    return cls


def get_default_hook(obj, default_hooks):
    """Finds the default save/load hook to use with the given object.

    Follows the Method Resolution Order, i.e., if no hook is registered for
    the class of the object itself, also searches classes which the object
    inherits from.

    Arguments
    ---------
    obj : instance
        Instance of a class.
    default_hooks : dict
        Mapping from classes to (checkpointing hook) functions.

    Returns
    -------
    The correct method or None if no method is registered.

    Example
    -------
    >>> a = paddle.nn.Layer()
    >>> get_default_hook(a, DEFAULT_SAVE_HOOKS) == torch_save
    True
    """
    mro = inspect.getmro(type(obj))
    for cls in mro:
        if cls in default_hooks:
            return default_hooks[cls]
    # If we got here, no hook found
    return None


Checkpoint = collections.namedtuple(
    "Checkpoint", ["path", "meta", "paramfiles"]
)
Checkpoint.__doc__ = """NamedTuple describing one saved checkpoint

To select a checkpoint to load from many checkpoint,
Checkpoints are first filtered and sorted based on this namedtuple.
Checkpointers put pathlib.Path in path and a dict in meta.
You can essentially add any info you want to meta when saving a checkpoint.
The only default key in meta is "unixtime".
Checkpoint.paramfiles is a dict from recoverable name to parameter filepath.
"""
# Creating a hash allows making checkpoint sets
Checkpoint.__hash__ = lambda self: hash(self.path)


def ckpt_recency(ckpt):
    """Recency as Checkpoint importance metric.

    This function can also act as an example of how to make checkpoint
    importance keyfuncs. This is a named function, but as you can see
    it could be easily implemented as a lambda in a pinch.
    """
    return ckpt.meta["unixtime"]


class Checkpointer:
    """Saves checkpoints and recovers from them.

    Arguments:

    checkpoints_dir : str, pathlib.Path
        Path to directory where to save checkpoints.
    recoverables : mapping, optional
        Objects to to recover. They need a (unique) name: this is used
        to connect the parameters in a checkpoint to the correct recoverable.
        The name is also used in the filename of the
        savefile for the objects parameters. These can also be added with
        add_recoverable or add_recoverables or just modifying
        checkpointer.recoverables directly.
    custom_load_hooks : mapping, optional
        A mapping from name [same as in recoverables] to function or method.
        Sets a custom loading hook for a particular object. The
        function/method must be callable with signature (instance, path)
        using positional arguments. This is satisfied by for example:
        `def loader(self, path)`.
    custom_save_hooks : mapping, optional
        Mapping from name [same as in recoverables] to function or method.
        Sets a custom saving hook for a particular object. The
        function/method must be callable with
        signature (instance, path) using positional arguments. This is
        satisfied by for example: def saver(self, path):
    allow_partial_load : bool, optional
        If True, allows loading a checkpoint where a savefile is not found
        for every registered recoverable. In that case, only the found
        savefiles are loaded. When False, loading such a save will raise
        RuntimeError. (default: False)

    Example
    -------
    >>> import paddle
    >>> #SETUP:
    >>> tempdir = getfixture('tmpdir')
    >>> class Recoverable(paddle.nn.Layer):
    ...     def __init__(self, param):
    ...         super().__init__()
    ...         self.param = torch.nn.Parameter(torch.tensor([param]))
    ...     def forward(self, x):
    ...         return x * self.param
    >>> recoverable = Recoverable(1.)
    >>> recoverables = {'recoverable': recoverable}
    >>> # SETUP DONE.
    >>> checkpointer = Checkpointer(tempdir, recoverables)
    >>> first_ckpt = checkpointer.save_checkpoint()
    >>> recoverable.param.data = torch.tensor([2.])
    >>> loaded_ckpt = checkpointer.recover_if_possible()
    >>> # Parameter has been loaded:
    >>> assert recoverable.param.data == torch.tensor([1.])
    >>> # With this call, by default, oldest checkpoints are deleted:
    >>> checkpointer.save_and_keep_only()
    >>> assert first_ckpt not in checkpointer.list_checkpoints()
    """

    def __init__(
        self,
        checkpoints_dir,
        recoverables=None,
        custom_load_hooks=None,
        custom_save_hooks=None,
        allow_partial_load=False,
    ):
        self.checkpoints_dir = pathlib.Path(checkpoints_dir)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self.recoverables = {}
        if recoverables is not None:
            self.add_recoverables(recoverables)
        self.custom_load_hooks = {}
        if custom_load_hooks is not None:
            self.custom_load_hooks.update(custom_load_hooks)
        self.custom_save_hooks = {}
        if custom_save_hooks is not None:
            self.custom_save_hooks.update(custom_save_hooks)
        self.allow_partial_load = allow_partial_load

    def add_recoverable(
        self, name, obj, custom_load_hook=None, custom_save_hook=None
    ):
        """Register a recoverable with possible custom hooks.

        Arguments
        ---------
        name : str
            Unique name for recoverable. Used to map savefiles to objects.
        obj : instance
            The object to recover.
        custom_load_hook : callable
            Called to load the object's savefile. The function/method must be
            callable with signature (instance, path) using positional
            arguments. This is satisfied by for example: def load(self, path):
        custom_save_hook : callable
            Called to save the object's parameters. The function/method must
            be callable with signature (instance, path) using positional
            arguments. This is satisfied by for example: def saver(self, path):
        """
        self.recoverables[name] = obj
        if custom_load_hook is not None:
            self.custom_load_hooks[name] = custom_load_hook
        if custom_save_hook is not None:
            self.custom_save_hooks[name] = custom_save_hook

    def add_recoverables(self, recoverables):
        """Update the recoverables dict from the given mapping.

        Arguments
        ---------
        recoverables : mapping
            Objects to recover.
            They need a (unique) name: this is used to
            connect the parameters in a checkpoint to the correct
            recoverable. The name is also used in the filename of the
            savefile for the objects parameters.
        """
        if isinstance(recoverables, collections.abc.Mapping):
            self.recoverables.update(recoverables)
        else:
            rec = repr(recoverables)  # noqa: F841, rec is used in MSG
            MSG = f"Checkpointer needs a mapping (e.g. dict), \
                    got {rec} instead."
            raise AttributeError(MSG)

    def save_checkpoint(
        self, meta={}, end_of_epoch=True, name=None, verbosity=logging.INFO
    ):
        """Saves a checkpoint.

        The whole checkpoint becomes a directory.
        Saves each registered object's parameters in a separate file.
        Also a meta file is added. The meta file by default has just the
        unixtime (seconds since unix epoch), but you can add anything
        relevant yourself. The meta information is later used to pick the
        checkpoint to load.

        The value of end_of_epoch is saved in the meta. This can affect how
        epoch counters and dataset iterators load their state.

        Arguments
        ---------
        meta : mapping, optional
            A mapping which is added to the meta file in the checkpoint. The
            key "unixtime" is included by default.
        end_of_epoch : bool, optional
            Whether the checkpoint is at the end of an epoch. True by default.
            May affect loading.
        name : str, optional
            Specify a custom name for your checkpoint.
            The name will still have a prefix added. If no name is given,
            a name is created from a timestamp and a random unique id.
        verbosity : logging level
            Set logging level this save.

        Returns
        -------
        Checkpoint
            namedtuple [see above], the saved checkpoint.
        """
        if name is None:
            ckpt_dir = self._new_checkpoint_dirpath()
        else:
            ckpt_dir = self._custom_checkpoint_dirpath(name)
        os.makedirs(ckpt_dir)  # May raise FileExistsError, let it.
        saved_meta = self._save_checkpoint_metafile(
            ckpt_dir / METAFNAME, meta, end_of_epoch
        )
        saved_paramfiles = {}
        for name, obj in self.recoverables.items():
            objfname = f"{name}" + PARAMFILE_EXT
            savepath = ckpt_dir / objfname
            saved_paramfiles[name] = savepath
            # First see if object has custom load hook:
            if name in self.custom_save_hooks:
                self.custom_save_hooks[name](obj, savepath)
                continue
            # Otherwise find the default saver for that type:
            default_hook = get_default_hook(obj, DEFAULT_SAVE_HOOKS)
            if default_hook is not None:
                # print("save path type: {}".format(type(str(savepath))))
                default_hook(obj, str(savepath))
                continue
            # If we got here, no custom hook or registered default hook
            MSG = f"Don't know how to save {type(obj)}. Register default hook \
                    or add custom hook for this object."
            raise RuntimeError(MSG)
        ckpt_type = "end-of-epoch" if end_of_epoch else "intra-epoch"
        logger.log(verbosity, f"Saved an {ckpt_type} checkpoint in {ckpt_dir}")
        return Checkpoint(ckpt_dir, saved_meta, saved_paramfiles)

    def save_and_keep_only(
        self,
        meta={},
        end_of_epoch=True,
        name=None,
        num_to_keep=1,
        keep_recent=True,
        importance_keys=[],
        max_keys=[],
        min_keys=[],
        ckpt_predicate=None,
        verbosity=logging.INFO,
    ):
        """Saves a checkpoint, then deletes the least important checkpoints.

        Essentially this combines ``save_checkpoint()`` and
        ``delete_checkpoints()`` in one call, providing short syntax.

        Arguments
        ---------
        meta : mapping, optional
            A mapping which is added to the meta file in the checkpoint. The
            key "unixtime" is included by default.
        end_of_epoch : bool, optional
            Whether the checkpoint is at the end of an epoch. True by default.
            May affect loading.
        name : str, optional
            Specify a custom name for your checkpoint.
            The name will still have a prefix added. If no name is given,
            a name is created from a timestamp and a random unique id.
        num_to_keep : int, optional
            Number of checkpoints to keep. Defaults to 1. This deletes all
            checkpoints remaining after filtering. Must be >=0.
        keep_recent : bool, optional
            Whether to keep the most recent ``num_to_keep`` checkpoints.
        importance_keys : list, optional
            A list of key functions used in sorting (see the sorted built-in).
            Each callable defines a sort order and num_to_keep checkpoints are
            kept for callable. The checkpoint with the highest keys are kept.
            The functions are passed Checkpoint namedtuples (see above).
        max_keys : list, optional
            A list of keys for which the *highest* value will be kept.
        min_keys : list, optional
            A list of keys for which the *lowest* value will be kept.
        ckpt_predicate : callable, optional
            Use this to exclude some checkpoints from deletion. Before any
            sorting, the list of checkpoints is filtered with this predicate.
            Only the checkpoints for which ckpt_predicate is True can be
            deleted. The function is called with Checkpoint namedtuples
            (see above).

        Returns
        -------
        None
            Unlike save_checkpoint, this does not return anything, since
            we cannot guarantee that the saved checkpoint actually survives
            deletion.
        """
        self.save_checkpoint(
            meta=meta, end_of_epoch=end_of_epoch, name=name, verbosity=verbosity
        )

        if keep_recent:
            importance_keys.append(ckpt_recency)
        self.delete_checkpoints(
            num_to_keep=num_to_keep,
            max_keys=max_keys,
            min_keys=min_keys,
            importance_keys=importance_keys,
            ckpt_predicate=ckpt_predicate,
            verbosity=verbosity,
        )

    def find_checkpoint(
        self,
        importance_key=None,
        max_key=None,
        min_key=None,
        ckpt_predicate=None,
    ):
        """Picks a particular checkpoint from all available checkpoints.

        If none of ``importance_key``, ``max_key``, and ``min_key`` is
        used, then most recent checkpoint will be returned. No more than
        one of them may be used.

        Most functionality is actually implemented in ``find_checkpoints()``
        but this is kept as a useful interface.

        Arguments
        ---------
        importance_key : callable, optional
            The key function used in sorting.
            The checkpoint with the highest returned value is picked.
            The function is called with Checkpoint namedtuples.
        max_key : str, optional
            The checkpoint with the highest value for this key will
            be returned. Only checkpoints with this key will be considered!
        min_key : str, optional
            The checkpoint with the lowest value for this key will
            be returned. Only checkpoints with this key will be considered!
        ckpt_predicate : callable, optional
            Before sorting, the list of
            checkpoints is filtered with this predicate.
            See the filter builtin.
            The function is called with Checkpoint namedtuples (see above).
            By default, all checkpoints are considered.

        Returns
        -------
        Checkpoint
            If found.
        None
            If no Checkpoints exist/remain after filtering.
        """
        ckpts_found = self.find_checkpoints(
            importance_key=importance_key,
            max_key=max_key,
            min_key=min_key,
            ckpt_predicate=ckpt_predicate,
            max_num_checkpoints=None,
        )
        if ckpts_found:
            return ckpts_found[0]
        else:
            return None

    def find_checkpoints(
        self,
        importance_key=None,
        max_key=None,
        min_key=None,
        ckpt_predicate=None,
        max_num_checkpoints=None,
    ):
        """Picks multiple checkpoints.

        If none of ``importance_key``, ``max_key``, and ``min_key`` is
        used, then the most recent checkpoints will be returned. No more than
        one of these may be used.

        Arguments
        ---------
        importance_key : callable, optional
            The key function used in sorting.
            The checkpoint with the highest returned value is picked.
            The function is called with Checkpoint namedtuples.
        max_key : str, optional
            The checkpoint with the highest value for this key will
            be returned. Only checkpoints with this key will be considered!
        min_key : str, optional
            The checkpoint with the lowest value for this key will
            be returned. Only checkpoints with this key will be considered!
        ckpt_predicate : callable, optional
            Before sorting, the list of
            checkpoints is filtered with this predicate.
            See the filter builtin.
            The function is called with Checkpoint namedtuples (see above).
            By default, all checkpoints are considered.
        max_num_checkpoints : int, None
            The maximum number of checkpoints to return, or None to return all
            found checkpoints.

        Returns
        -------
        list
            List containing at most the max specified number of Checkpoints.

        """
        if importance_key is None and min_key is None and max_key is None:
            importance_key = ckpt_recency

        if max_key and not importance_key:

            def importance_key(ckpt):
                return ckpt.meta[max_key]

            def ckpt_predicate(ckpt, old_predicate=ckpt_predicate):
                if old_predicate is not None:
                    return max_key in ckpt.meta and old_predicate(ckpt)
                else:
                    return max_key in ckpt.meta

        elif min_key and not importance_key:

            def importance_key(ckpt):
                return -ckpt.meta[min_key]

            def ckpt_predicate(ckpt, old_predicate=ckpt_predicate):
                if old_predicate is not None:
                    return min_key in ckpt.meta and old_predicate(ckpt)
                else:
                    return min_key in ckpt.meta

        elif min_key or max_key:
            raise ValueError(
                "Must specify only one of 'importance_key', 'max_key', "
                "and 'min_key'."
            )

        ckpts = self.list_checkpoints()
        ckpts = list(filter(ckpt_predicate, ckpts))
        # First sort by recency, so that importance being equal,
        # the most checkpoints are returned
        ckpts = sorted(ckpts, key=ckpt_recency, reverse=True)
        if ckpts:
            ranked_ckpts = sorted(ckpts, key=importance_key, reverse=True)
            # NOTE: apparently, you can also slice [:None],
            # and this is the same as [:], so the following if-else is not
            # strictly speaking needed. However, this feature does not seem to
            # be documented Python so I don't want to trust it.
            if max_num_checkpoints is not None:
                return ranked_ckpts[:max_num_checkpoints]
            else:  # No max number -> return all ckpts, but just sorted
                return ranked_ckpts
        else:
            return []  # Be explicit :)

    def recover_if_possible(
        self,
        importance_key=None,
        max_key=None,
        min_key=None,
        ckpt_predicate=None,
        device=None,
    ):
        """Picks a checkpoint and recovers from that, if one is found.

        If a checkpoint is not found, no recovery is run.

        If none of ``importance_key``, ``max_key``, and ``min_key`` is
        used, then most recent checkpoint will be returned. No more than
        one of them may be used.

        Arguments
        ---------
        importance_key : callable, optional
            The key function used in sorting.
            The checkpoint with the highest returned value is loaded.
            The function is called with Checkpoint namedtuples.
        max_key : str, optional
            The checkpoint with the highest value for this key will be loaded.
            Only checkpoints with this key will be considered!
        min_key : str, optional
            The checkpoint with the lowest value for this key will be loaded.
            Only checkpoints with this key will be considered!
        ckpt_predicate : callable, optional
            Before sorting, the list of
            checkpoints is filtered with this predicate.
            See the filter builtin.
            The function is called with Checkpoint namedtuples (see above).
            By default, all checkpoints are considered.
        device : torch.device
            Device to load models to.

        Returns
        -------
        Checkpoint
            If found.
        None
            If no Checkpoints exist/remain after filtering.
        """
        chosen_ckpt = self.find_checkpoint(
            importance_key, max_key, min_key, ckpt_predicate,
        )
        if chosen_ckpt is not None:
            self.load_checkpoint(chosen_ckpt, device)
        else:
            logger.info("Would load a checkpoint here, but none found yet.")
        return chosen_ckpt

    def load_checkpoint(self, checkpoint, device=None):
        """Loads the specified checkpoint.

        Arguments
        ---------
        checkpoint : Checkpoint
            Checkpoint to load.
        """
        self._call_load_hooks(checkpoint, device)

    def list_checkpoints(self):
        """List all checkpoints in the checkpoints directory.

        Returns
        -------
        list
            List of Checkpoint namedtuple (see above).
        """
        return self._construct_checkpoint_objects(self._list_checkpoint_dirs())

    # NOTE: * in arglist -> keyword only arguments
    def delete_checkpoints(
        self,
        *,
        num_to_keep=1,
        min_keys=None,
        max_keys=None,
        importance_keys=[ckpt_recency],
        ckpt_predicate=None,
        verbosity=logging.INFO,
    ):
        """Deletes least important checkpoints.

        Since there can be many ways to define importance (e.g. lowest WER,
        lowest loss), the user should provide a list of sort key functions,
        each defining a particular importance order. In essence, each
        importance key function extracts one importance metric (higher is more
        important). For each of these orders, num_to_keep checkpoints are kept.
        However if there is overlap between each orders' preserved checkpoints,
        the additional checkpoints are not preserved, so the total number of
        preserved checkpoints can be less than::

            num_to_keep * len(importance_keys)

        Arguments
        ---------
        num_to_keep : int, optional
            Number of checkpoints to keep.
            Defaults to 10. You choose to keep 0. This deletes all
            checkpoints remaining after filtering. Must be >=0
        min_keys : list, optional
            List of strings representing keys in the meta. The lowest of
            these values will be kept, up to num_to_keep.
        max_keys : list, optional
            List of strings representing keys in the meta. The highest of
            these values will be kept, up to num_to_keep.
        importance_keys : list, optional
            A list of key functions used in sorting (see the sorted built-in).
            Each callable defines a sort order and num_to_keep checkpoints are
            kept for  callable. To be clear, those with the highest key are
            kept.
            The functions are called with Checkpoint namedtuples
            (see above). See also the default (ckpt_recency,
            above). The default deletes all but the latest checkpoint.
        ckpt_predicate : callable, optional
            Use this to exclude some checkpoints from deletion. Before any
            sorting, the list of checkpoints is filtered with this predicate.
            Only the checkpoints for which ckpt_predicate is True can be
            deleted. The function is called with Checkpoint namedtuples
            (see above).
        verbosity : logging level
            Set logging level for this deletion.

        Note
        ----
        Must be called with keyword arguments, as a signoff that you
        know what you are doing. Deletion is permanent.
        """
        if num_to_keep < 0:
            raise ValueError("Number of checkpoints to keep must be positive.")

        # Build a list of potential deletions and protected checkpoints
        potential_deletions = set()
        protected_checkpoints = set()
        keys = [{"min_key": key} for key in min_keys or []]
        keys.extend([{"max_key": key} for key in max_keys or []])
        keys.extend([{"importance_key": key} for key in importance_keys])

        # Don't consider checkpoints for deletion that don't have a listed key
        for key_kwargs in keys:
            key_kwargs["ckpt_predicate"] = ckpt_predicate
            potential_deletions.update(self.find_checkpoints(**key_kwargs))
            protected_checkpoints.update(
                self.find_checkpoints(
                    max_num_checkpoints=num_to_keep, **key_kwargs
                )
            )

        # Delete unprotected checkpoints
        for ckpt in potential_deletions:
            if ckpt not in protected_checkpoints:
                Checkpointer._delete_checkpoint(ckpt, verbosity=verbosity)

    @staticmethod
    def _delete_checkpoint(checkpoint, verbosity=logging.INFO):
        if not Checkpointer._is_checkpoint_dir(checkpoint.path):
            raise RuntimeError("Checkpoint does not appear valid for deletion.")
        shutil.rmtree(checkpoint.path)
        logger.log(verbosity, f"Deleted checkpoint in {checkpoint.path}")

    def _call_load_hooks(self, checkpoint, device=None):
        # This internal function finds the correct hook to call for every
        # recoverable, and calls it.
        logger.info(f"Loading a checkpoint from {checkpoint.path}")
        end_of_epoch = checkpoint.meta["end-of-epoch"]
        for name, obj in self.recoverables.items():
            # NOTE: We want the checkpoint namedtuple to have the paramfile
            # paths for each recoverable.
            # In some rare case, the user can e.g. add a path there manually.
            try:
                loadpath = checkpoint.paramfiles[name]
            except KeyError:
                if self.allow_partial_load:
                    continue
                elif "dataloader" in name:
                    MSG = f"Loading checkpoint from {checkpoint.path}, \
                            but missing a load path for {name}"
                    warnings.warn(MSG, UserWarning)
                    continue
                else:
                    MSG = f"Loading checkpoint from {checkpoint.path}, \
                            but missing a load path for {name}"
                    raise RuntimeError(MSG)

            # First see if object has custom load hook:
            if name in self.custom_load_hooks:
                self.custom_load_hooks[name](
                    obj, loadpath, end_of_epoch, device
                )
                continue
            # Otherwise find the default saver for that type:
            default_hook = get_default_hook(obj, DEFAULT_LOAD_HOOKS)
            if default_hook is not None:
                default_hook(obj, loadpath, end_of_epoch, device)
                continue
            # If we got here, no custom hook or registered default hook exists
            MSG = f"Don't know how to load {type(obj)}. Register default hook \
                    or add custom hook for this object."
            raise RuntimeError(MSG)

    def _list_checkpoint_dirs(self):
        # This internal method returns a list of individual checkpoint
        # directory paths in the top checkpoint directory
        return [
            x
            for x in self.checkpoints_dir.iterdir()
            if Checkpointer._is_checkpoint_dir(x)
        ]

    @staticmethod
    def _construct_checkpoint_objects(checkpoint_dirs):
        # This internal method takes a list of individual checkpoint
        # directory paths (as produced by _list_checkpoint_dirs)
        checkpoints = []
        for ckpt_dir in checkpoint_dirs:
            with open(ckpt_dir / METAFNAME) as fi:
                meta = yaml.load(fi, Loader=yaml.Loader)
            paramfiles = {}
            for ckptfile in ckpt_dir.iterdir():
                if ckptfile.suffix == PARAMFILE_EXT:
                    paramfiles[ckptfile.stem] = ckptfile
            checkpoints.append(Checkpoint(ckpt_dir, meta, paramfiles))
        return checkpoints

    @staticmethod
    def _is_checkpoint_dir(path):
        # This internal method verifies whether a given path points to a
        # directory that holds a checkpoint.
        path = pathlib.Path(path)
        if not path.is_dir():
            return False
        if not path.name.startswith(CKPT_PREFIX):
            return False
        return (path / METAFNAME).exists()

    def _new_checkpoint_dirpath(self):
        # This internal method creates a checkpoint name and returns a path
        # to that directory (but does not create the directory!)
        t = time.time()
        stamp = time.strftime("%Y-%m-%d+%H-%M-%S", time.localtime(t))
        suffix_num = 0
        while (
            self.checkpoints_dir / f"{CKPT_PREFIX}+{stamp}+{suffix_num:02d}"
        ).exists():
            suffix_num += 1
        return self.checkpoints_dir / f"{CKPT_PREFIX}+{stamp}+{suffix_num:02d}"

    def _custom_checkpoint_dirpath(self, name):
        # This internal method creates a checkpoint name based on a given
        # custom name and returns a path to that directory (but does not
        # create the directory!)
        return self.checkpoints_dir / f"{CKPT_PREFIX}+{name}"

    def _save_checkpoint_metafile(
        self, fpath, meta_to_include={}, end_of_epoch=True
    ):
        # This internal method saves the meta information in the given path
        meta = {"unixtime": time.time(), "end-of-epoch": end_of_epoch}
        meta.update(meta_to_include)
        with open(fpath, "w") as fo:
            fo.write("# yamllint disable\n")
            fo.write(yaml.dump(meta))
        return meta


def average_state_dicts(state_dicts):
    """Produces an average state_dict from an iterator over state_dicts.

    Note that at one time, this keeps two of the state_dicts in memory, which
    is the minimum memory requirement.

    Arguments
    ---------
    state_dicts : iterator, list
        The state_dicts to average.

    Returns
    -------
    state_dict
        The averaged state_dict.
    """
    iterator = iter(state_dicts)
    try:
        running_sum = next(iterator)
    except StopIteration:
        raise ValueError("No state dicts to average.")
    num_dicts = 1
    with torch.no_grad():
        # First sum all state_dicts together:
        for state_dict in iterator:
            for pname, param in state_dict.items():
                running_sum[pname] += param.data
            num_dicts += 1
        # Finally, divide by number of dicts:
        for pname, param in running_sum.items():
            running_sum[pname] = param.data / float(num_dicts)
    return running_sum


def average_checkpoints(
    checkpoint_list,
    recoverable_name,
    parameter_loader=paddle.load,
    averager=average_state_dicts,
    device=None,
):
    """Average parameters from multiple checkpoints.

    Use Checkpointer.find_checkpoints() to get the list of checkpoints to
    average over.
    Averaging parameters from some of the last checkpoints in training has been
    shown to sometimes improve performance.

    The default loader and averager work for standard PyTorch modules.

    Arguments
    ---------
    checkpoint_list : list
        List of checkpoints to average.
    recoverable_name : str
        The name of the recoverable, the parameters of which are loaded and
        averaged.
    parameter_loader : function
        A function which takes a single argument, the path to a parameter file,
        and loads the parameters from that file. By default, torch.load,
        which produces state_dict dictionaries.
    averager : function
        A function which takes an iterator over the parameters from each
        checkpoint, as loaded by parameter_loader, and produces their average.
        Note that the function is called with an iterator, so the length is
        initially unknown; the implementation should simply count the number of
        different parameter sets as they are yielded. See average_state_dicts
        above for an example. It is the default averager, and averages
        state_dicts.

    Returns
    -------
    Any
        The output of the averager function.

    Example
    -------
    >>> # Consider this toy Module again:
    >>> class Recoverable(paddle.nn.Layer):
    ...     def __init__(self, param):
    ...         super().__init__()
    ...         self.param = torch.nn.Parameter(torch.tensor([param]))
    ...     def forward(self, x):
    ...         return x * self.param
    >>> # Now let's make some checkpoints:
    >>> model = Recoverable(1.)
    >>> tempdir = getfixture('tmpdir')
    >>> checkpointer = Checkpointer(tempdir, {"model": model})
    >>> for new_param in range(10):
    ...     model.param.data = torch.tensor([float(new_param)])
    ...     _ = checkpointer.save_checkpoint()  # Suppress output with assignment
    >>> # Let's average the 3 latest checkpoints
    >>> # (parameter values 7, 8, 9 -> avg=8)
    >>> ckpt_list = checkpointer.find_checkpoints(max_num_checkpoints = 3)
    >>> averaged_state = average_checkpoints(ckpt_list, "model")
    >>> # Now load that state in the normal way:
    >>> _ = model.load_state_dict(averaged_state)  # Suppress output
    >>> model.param.data
    tensor([8.])
    """

    try:
        # try to map the ckps to the correct device
        parameter_iterator = (
            parameter_loader(
                ckpt.paramfiles[recoverable_name], map_location=device
            )
            for ckpt in checkpoint_list
        )
    except TypeError:
        parameter_iterator = (
            parameter_loader(ckpt.paramfiles[recoverable_name])
            for ckpt in checkpoint_list
        )
    return averager(parameter_iterator)
