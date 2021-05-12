#!/usr/bin/env python3


def has_module(module):
    try:
        __import__(module)
        return True
    except ImportError:
        pass
