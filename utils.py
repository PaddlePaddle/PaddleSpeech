"""Contains common utility functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def print_arguments(args):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    print("-----  Configuration Arguments -----")
    for arg, value in vars(args).iteritems():
        print("%s: %s" % (arg, value))
    print("------------------------------------")
