#!/usr/bin/env python

# MIT License
#
# Copyright (c) 2018, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S. Dept. of Energy).  All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

from __future__ import absolute_import
from functools import wraps
import functools

from .plot import *
from .options import *
from .mpi_support import *

__all__ = [] # import for side effects


#------------------------------------------------------------------------------#
class auto_timer(object):
    """ A decorator for the auto-timer, e.g.:
        @tim.util.auto_timer("'AUTO_TIMER_DECORATOR_KEY_TEST':{}".format(tim.LINE()))
        def main(n):
            for i in range(2):
                fibonacci(n * (i+1))
        # ...
    """
    def __init__(self, key="", add_args=False):
        self.key = key
        self.add_args = add_args


    def __call__(self, func):

        import timemory as tim
        file = tim.FILE(3)
        line = tim.LINE(2)

        @wraps(func)
        def function_wrapper(*args, **kwargs):

            # add_args only if key not specified
            if self.add_args and self.key == "":
                self.key = self.arg_string(args, kwargs)

            t = tim.timer_decorator(func.__name__, file, line,
                self.key, self.add_args)

            return func(*args, **kwargs)

        return function_wrapper


    def arg_string(self, args, kwargs):
        str = '('
        for i in range(0, len(args)):
            if i == 0:
                str = '{}{}'.format(str, args[i])
            else:
                str = '{}, {}'.format(str, args[i])

        for key, val in kwargs:
            str = '{}, {}={}'.format(str, key, val)

        return '{})'.format(str)

