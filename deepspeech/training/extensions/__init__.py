
from typing import Callable

from .extension import Extension

def make_extension(trigger: Callable=None,
                   default_name: str=None,
                   priority: int=None,
                   finalizer: Callable=None,
                   initializer: Callable=None,
                   on_error: Callable=None):
    """Make an Extension-like object by injecting required attributes to it.
    """
    if trigger is None:
        trigger = Extension.trigger
    if priority is None:
        priority = Extension.priority

    def decorator(ext):
        ext.trigger = trigger
        ext.default_name = default_name or ext.__name__
        ext.priority = priority
        ext.finalize = finalizer
        ext.on_error = on_error
        ext.initialize = initializer
        return ext

    return decorator