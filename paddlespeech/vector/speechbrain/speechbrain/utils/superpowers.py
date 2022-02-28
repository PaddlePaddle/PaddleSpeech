"""Superpowers which should be sparingly used.

This library contains functions for importing python files and
for running shell commands. Remember, with great power comes great
responsibility.

Authors
 * Mirco Ravanelli 2020
 * Aku Rouhe 2021
"""

import logging
import subprocess
import importlib
import pathlib

logger = logging.getLogger(__name__)


def import_from_path(path):
    r"""Import module from absolute path

    Arguments
    ---------
    path : str, pathlib.Path
        The path to the module to import

    Returns
    -------
    module
        The loaded module

    >>> modulepath = getfixture("tmpdir") / "helloer.py"
    >>> with open(modulepath, "w") as fo:
    ...     _ = fo.write("def a():\n\treturn 'hello'")
    >>> helloer = import_from_path(modulepath)
    >>> helloer.a()
    'hello'

    Implementation taken from:
    https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    """
    path = pathlib.Path(path)
    modulename = path.with_suffix("").name
    spec = importlib.util.spec_from_file_location(modulename, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_shell(cmd):
    r"""This function can be used to run a command in the bash shell.

    Arguments
    ---------
    cmd : str
        Shell command to run.

    Returns
    -------
    bytes
        The captured standard output.
    bytes
        The captured standard error.
    int
        The returncode.

    Raises
    ------
    OSError
        If returncode is not 0, i.e., command failed.

    Example
    -------
    >>> out, err, code = run_shell("echo 'hello world'")
    >>> out.decode(errors="ignore")
    'hello world\n'
    """

    # Executing the command
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )

    # Capturing standard output and error
    (output, err) = p.communicate()

    if p.returncode != 0:
        raise OSError(err.decode(errors="replace"))

    # Adding information in the logger
    msg = output.decode(errors="replace") + "\n" + err.decode(errors="replace")
    logger.debug(msg)

    return output, err, p.returncode
