"""Convenience functions for the simplest parameter transfer cases.

Use `speechbrain.utils.checkpoints.Checkpointer` to find a checkpoint
and the path to the parameter file.

Authors
 * Aku Rouhe 2020
"""
import logging
import pathlib

from speechbrain.pretrained.fetching import fetch
from speechbrain.utils.checkpoints import (
    DEFAULT_LOAD_HOOKS,
    DEFAULT_TRANSFER_HOOKS,
    PARAMFILE_EXT,
    get_default_hook,
)

logger = logging.getLogger(__name__)


class Pretrainer:
    """Orchestrates pretraining

    First collects parameter file symlinks into the given directory. Then
    calls load hooks for each of those parameter files.

    Arguments
    ---------
    collect_in : str or Path
        Path to directory where the parameter file symlinks are collected.
    loadables : mapping
        Mapping from loadable key to object. This connects the keys to
        the actual object instances.
    paths : mapping
        Mapping from loadable key to filepath. The last part
        of the path is treated as file name, the rest of it
        is treated as a "source" which can be either a directory
        path or a magic source like Huggingface hub ID.
        e.g. sb/asr-crdnn-libri/lm.ckpt
        -> source=sb/asr-crdnn-libri, file=lm.ckpt
        Note that when collecting, you can specify a default source,
        which is used for all loadables that don't have a path specified.
    custom_hooks : mapping
        Mapping from loadable key to parameter transfer hook function. If you
        want to use a custom loading function, specify it here.
    """

    def __init__(
        self,
        collect_in="./model_checkpoints",
        loadables=None,
        paths=None,
        custom_hooks=None,
    ):
        self.loadables = {}
        self.collect_in = pathlib.Path(collect_in)
        if loadables is not None:
            self.add_loadables(loadables)
        self.paths = {}
        if paths is not None:
            self.add_paths(paths)
        self.custom_hooks = {}
        if custom_hooks is not None:
            self.add_custom_hooks(custom_hooks)

    def set_collect_in(self, path):
        """Change the collecting path"""
        self.collect_in = pathlib.Path(path)

    def add_loadables(self, loadables):
        """Update the loadables dict from the given mapping.

        Arguments
        ---------
        loadables : mapping
            Mapping from loadable key to object
        """
        self.loadables.update(loadables)

    def add_paths(self, paths):
        """Update the paths for different loadables.

        When collecting parameters, paths here are preferred. Note that when
        collecting, you can specify a default source, which is used for all
        loadables that don't have a path specified.

        Arguments
        ---------
        paths : mapping
            Mapping from loadable key to filepath. The last part
            of the path is treated as file name, the rest of it
            is treated as a "source" which can be either a directory
            path or a magic source like Huggingface hub ID.
            e.g. sb/asr-crdnn-libri/lm.ckpt
            -> source=sb/asr-crdnn-libri, file=lm.ckpt
        """
        self.paths.update(paths)

    def add_custom_hooks(self, custom_hooks):
        """Update the custom hooks.

        When loading parameters, hooks here are preferred over class defaults.

        Arguments
        ---------
        custom_hooks : mapping
            Mapping from loadable key to parameter transfer hook function. If
            you want to use a custom loading function, specify it here.

        """
        self.custom_hooks.update(custom_hooks)

    @staticmethod
    def split_path(path):
        """Splits a path to source and filename

        This also handles URLs and Huggingface hub paths, in addition to
        regular paths.

        Arguments
        ---------
        path : str

        Returns
        -------
        str
            Source
        str
            Filename
        """
        if "/" in path:
            return path.rsplit("/", maxsplit=1)
        else:
            # Interpret as path to file in current directory.
            return "./", path

    def collect_files(self, default_source=None):
        """Fetches parameters from known paths with fallback default_source

        The actual parameter files may reside elsewhere, but this ensures a
        symlink in the self.collect_in directory. The symlink always uses the
        loadable key in the filename. This standardization makes it easier to
        orchestrate pretraining on e.g. distributed setups.

        Use the default_source if you have everything organized neatly into one
        location, like a Huggingface hub repo.

        Arguments
        ---------
        default_source : str or Path
            This is used for each loadable which doesn't have a path already
            specified. If the loadable has key "asr", then the file to look for is
            default_source/asr.ckpt

        Returns
        -------
        dict
            Mapping from loadable key to a local path from which loadable's
            parameters can be loaded. This is not used in this class, but
            can possibly be helpful.
        """
        logger.debug(
            f"Collecting files (or symlinks) for pretraining in {self.collect_in}."
        )
        self.collect_in.mkdir(exist_ok=True)
        loadable_paths = {}
        # print("loadables: {}".format(self.loadables))
        for name in self.loadables:
            save_filename = name + PARAMFILE_EXT
            if name in self.paths:
                source, filename = self.split_path(self.paths[name])
            elif default_source is not None:
                filename = save_filename
                source = default_source
            else:
                raise ValueError(
                    f"Path not specified for '{name}', "
                    "and no default_source given!"
                )
            # print("filename, source: {}, {}".format(filename, source))
            path = fetch(
                filename, source, self.collect_in, save_filename=save_filename
            )
            loadable_paths[name] = path
        return loadable_paths

    def load_collected(self, device=None):
        """Loads the files that have been collected.

        Arguments
        ---------
        device : str
            Device on which to load, if you want to load to a specific device
            directly ( otherwise just leave it to None ).
        """
        logger.info(
            f"Loading pretrained files for: {', '.join(self.loadables)}"
        )
        paramfiles = {}
        for name in self.loadables:
            filename = name + PARAMFILE_EXT
            paramfiles[name] = self.collect_in / filename
        print("load hooks: {}, {}".format(paramfiles, device))
        self._call_load_hooks(paramfiles, device)

    def _call_load_hooks(self, paramfiles, device=None):
        # This internal function finds the correct hook to call for every
        # recoverable, and calls it.
        for name, obj in self.loadables.items():
            loadpath = paramfiles[name]

            # First see if object has custom load hook:
            if name in self.custom_hooks:
                self.custom_hooks[name](obj, loadpath, )
                continue
            # Try the default transfer hook:
            default_hook = get_default_hook(obj, DEFAULT_TRANSFER_HOOKS)
            if default_hook is not None:
                print("default hook: {}".format(default_hook))
                default_hook(obj, loadpath, device=device)
                continue
            # Otherwise find the default loader for that type:
            default_hook = get_default_hook(obj, DEFAULT_LOAD_HOOKS)
            if default_hook is not None:
                # Need to fake end-of-epoch:
                end_of_epoch = False
                default_hook(obj, loadpath, end_of_epoch, device)
                continue
            # If we got here, no custom hook or registered default hook exists
            MSG = f"Don't know how to load {type(obj)}. Register default hook \
                    or add custom hook for this object."
            raise RuntimeError(MSG)
