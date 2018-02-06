#!/usr/bin/env python
#
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

## @file util.py
## Decorators for TiMemory module
##

from __future__ import absolute_import

import os
import sys


#------------------------------------------------------------------------------#
class base_decorator(object):
    """
    A base class for the decorators
    """
    def __init__(self, key="", add_args=False, is_class=False):
        self.key = key
        self.add_args = add_args
        self.is_class = is_class


    # ------------------------------------------------------------------------ #
    def class_string(self, args, kwargs):
        """
        Generate a class identifier
        """
        _str = ''
        if self.is_class and len(args) > 0 and args[0] is not None:
            _str = '[{}]'.format(type(args[0]).__name__)
            # this check guards against old bug from class methods
            # calling class methods
            if not _str in self.key:
                return _str
            else:
                return ''
        return _str


    # ------------------------------------------------------------------------ #
    def arg_string(self, args, kwargs):
        """
        Generate a string of the arguments
        """
        _str = '{}'.format(self.class_string(args, kwargs))
        if self.add_args:
            _str = '{}('.format(_str)
            for i in range(0, len(args)):
                if i == 0:
                    _str = '{}{}'.format(_str, args[i])
                else:
                    _str = '{}, {}'.format(_str, args[i])

            for key, val in kwargs:
                _str = '{}, {}={}'.format(_str, key, val)

            return '{})'.format(_str)
        return _str


#------------------------------------------------------------------------------#
class auto_timer(base_decorator):
    """ A decorator for the auto-timer, e.g.:
        @timemory.util.auto_timer(add_args=True)
        def main(n=5):
            for i in range(2):
                fibonacci(n * (i+1))
        # ...
        # output :
        # > [pyc] main(5)@'example.py':10 ...
    """
    # ------------------------------------------------------------------------ #
    def __init__(self, key="", add_args=False, is_class=False, report_at_exit=False):
        super(auto_timer, self).__init__(key, add_args, is_class)
        self.report_at_exit = report_at_exit


    # ------------------------------------------------------------------------ #
    def __call__(self, func):

        import timemory
        from functools import wraps
        _file = timemory.FILE(3)
        _line = timemory.LINE(2)

        @wraps(func)
        def function_wrapper(*args, **kwargs):

            _key = '{}{}'.format(self.key, self.arg_string(args, kwargs))

            t = timemory.timer_decorator(func.__name__, _file, _line,
                _key, self.add_args or self.is_class, self.report_at_exit)

            return func(*args, **kwargs)

        return function_wrapper


#------------------------------------------------------------------------------#
class timer(base_decorator):
    """ A decorator for the timer, e.g.:

        class someclass(object):

            @timemory.util.timer(is_class=True)
            def __init__(self):
                self.some_obj = None
        # ...
    """
    # ------------------------------------------------------------------------ #
    def __init__(self, key="", add_args=False, is_class=False):
        super(timer, self).__init__(key, add_args, is_class)


    # ------------------------------------------------------------------------ #
    def __call__(self, func):

        import timemory
        from functools import wraps
        _file = timemory.FILE(3)
        _line = timemory.LINE(2)

        @wraps(func)
        def function_wrapper(*args, **kwargs):

            _key = ''
            _args = self.arg_string(args, kwargs)
            if self.key == "":
                _func = func.__name__
                _key = '{}{}@{}{}'.format(_func, _args, _file, _line)
            else:
                _key = '{}{}'.format(self.key, _args)

            t = timemory.timer(_key)

            t.start()
            ret = func(*args, **kwargs)
            t.stop()
            t.report()
            return ret

        return function_wrapper


#------------------------------------------------------------------------------#
class rss_usage(base_decorator):
    """ A decorator for the rss usage, e.g.:

        class someclass(object):

            @timemory.util.rss_usage(is_class=True)
            def __init__(self):
                self.some_obj = None
        # ...
    """
    # ------------------------------------------------------------------------ #
    def __init__(self, key="", add_args=False, is_class=False):
        super(rss_usage, self).__init__(key, add_args, is_class)


    # ------------------------------------------------------------------------ #
    def __call__(self, func):

        import timemory
        from functools import wraps
        _file = timemory.FILE(3)
        _line = timemory.LINE(2)

        @wraps(func)
        def function_wrapper(*args, **kwargs):

            _key = ''
            _args = self.arg_string(args, kwargs)
            if self.key == "":
                _func = func.__name__
                _key = '{}{}@{}{}'.format(_func, _args, _file, _line)
            else:
                _key = '{}{}'.format(self.key, _args)

            # initial values
            rss_done = timemory.rss_usage()
            rss_self = timemory.rss_usage()
            # record pre-function
            rss_self.record()
            # run function
            ret = func(*args, **kwargs)
            # record at end
            rss_done.record()
            # calc self
            rss_self = rss_done - rss_self

            print('{} : RSS {}total,self{}_{}current,peak{} : ({}|{}) | ({}|{}) [MB]'.format(
                _key, '{', '}', '{', '}',
                rss_done.current(), rss_done.peak(),
                rss_self.current(), rss_self.peak()))

            return ret

        return function_wrapper
