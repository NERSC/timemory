#!/usr/bin/env python

from __future__ import absolute_import

from .plot import *
from .options import *
from .mpi_support import *

__all__ = [] # import for side effects

def decorate_auto_timer(func, key = ""):
    import timemory as tim
    file = tim.FILE(3)
    line = tim.LINE(2)
    def function_wrapper(*args, **kwargs):
        t = tim.timer_decorator(func.__name__, file, line, key)
        res = func(*args, **kwargs)
        return res
    return function_wrapper
