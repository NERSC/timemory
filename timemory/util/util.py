#!@PYTHON_EXECUTABLE@
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

# @file util.py
# Decorators for TiMemory module
#

from __future__ import absolute_import
import os
import six
import sys
import inspect
from enum import Enum
from functools import wraps

from ..common import FILE, FUNC, LINE
from ..libpytimemory import settings, timer_decorator, component_decorator
from ..libpytimemory import timer, rss_usage, component

__author__ = "Jonathan Madsen"
__copyright__ = "Copyright 2020, The Regents of the University of California"
__credits__ = ["Jonathan Madsen"]
__license__ = "MIT"
__version__ = "@PROJECT_VERSION@"
__maintainer__ = "Jonathan Madsen"
__email__ = "jrmadsen@lbl.gov"
__status__ = "Development"
__all__ = ['base_decorator',
           'auto_timer',
           'auto_tuple',
           'timer',
           'rss_usage',
           'marker']


#----------------------------------------------------------------------------------------#
class context(Enum):
    blank = 0
    basic = 1
    full = 2
    defer = 3

#----------------------------------------------------------------------------------------#
#


class base_decorator(object):
    """
    A base class for the decorators and context managers
    """

    #------------------------------------------------------------------------------------#
    #
    def __init__(self, key="", add_args=False, is_class=False, mode="defer"):
        self.key = key
        self.add_args = add_args
        self.is_class = is_class
        self.signature = context.defer
        self.mode = mode
        if self.mode != "defer":
            try:
                self.signature = getattr(context, self.mode)
            except Exception as e:
                pass

    #------------------------------------------------------------------------------------#
    #
    def determine_signature(self, is_decorator=True, is_context_manager=False):

        def determine_deferred(_is_decorate, _is_context):
            if _is_decorate:
                return context.basic
            elif _is_context:
                return context.blank
            else:
                return context.full

        if self.mode != "defer":
            try:
                self.signature = getattr(context, self.mode)
            except Exception as e:
                print(e)
                self.signature = determine_deferred(
                    is_decorator, is_context_manager)
        elif self.signature == context.defer:
            self.signature = determine_deferred(
                is_decorator, is_context_manager)

    #------------------------------------------------------------------------------------#
    #
    def parse_wrapped(self, func, args, kwargs):

        if len(args) > 0 and args[0] is not None and inspect.isclass(type(args[0])):
            self.is_class = True
        else:
            self.is_class = False

    #------------------------------------------------------------------------------------#
    #
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
    #------------------------------------------------------------------------------------#
    #

    def arg_string(self, args, kwargs):
        """
        Generate a string of the arguments
        """
        #_str = '{}'.format(self.class_string(args, kwargs))
        _str = ""
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


#----------------------------------------------------------------------------------------#
#
class auto_timer(base_decorator):
    """ A decorator or context-manager for the auto-timer, e.g.:
        @timemory.util.auto_timer(add_args=True)
        def main(n=5):
            for i in range(2):
                fibonacci(n * (i+1))
        # ...
        # output :
        # > [pyc] main(5)@'example.py':10 ...
    """

    #------------------------------------------------------------------------------------#
    #
    def __init__(self, key="", add_args=False, is_class=False, report_at_exit=False,
                 mode="defer"):
        super(auto_timer, self).__init__(
            key=key, add_args=add_args, is_class=is_class, mode=mode)
        self.report_at_exit = report_at_exit
        self._self_obj = None
    #------------------------------------------------------------------------------------#
    #

    def __call__(self, func):
        """
        Decorator
        """
        _file = FILE(3)
        _line = LINE(2)

        @wraps(func)
        def function_wrapper(*args, **kwargs):
            self.parse_wrapped(func, args, kwargs)
            self.determine_signature(
                is_decorator=True, is_context_manager=False)

            _func = func.__name__
            _key = ''
            _args = self.arg_string(args, kwargs)
            if self.signature == context.blank:
                _key = '{}{}'.format(self.key, _args)
            elif self.signature == context.basic:
                _key = '{}/{}/{}'.format(_func, self.key, _args)
            elif self.signature == context.full:
                _key = '{}/{}:{}/{}{}'.format(
                    _func, _file, _line, self.key, _args)
            _key = _key.strip('/')

            _dec = timer_decorator(_key, self.report_at_exit)
            _ret = func(*args, **kwargs)
            del _dec
            return _ret

        return function_wrapper
    #------------------------------------------------------------------------------------#
    #

    def __enter__(self, *args, **kwargs):
        """
        Context manager
        """
        _file = FILE(3)
        _line = LINE(2)
        _func = FUNC(2)
        self.determine_signature(is_decorator=False, is_context_manager=True)

        _key = ''
        _args = self.arg_string(args, kwargs)
        if self.signature == context.blank:
            _key = '{}{}'.format(self.key, _args)
        elif self.signature == context.basic:
            _key = '{}/{}/{}'.format(_func, self.key, _args)
        elif self.signature == context.full:
            _key = '{}/{}:{}/{}{}'.format(
                _func, _file, _line, self.key, _args)
        _key = _key.strip('/')

        self._self_obj = timer_decorator(_key, self.report_at_exit)
    #------------------------------------------------------------------------------------#
    #

    def __exit__(self, exc_type, exc_value, exc_traceback):
        del self._self_obj

        if exc_type is not None and exc_value is not None and exc_traceback is not None:
            import traceback
            traceback.print_exception(
                exc_type, exc_value, exc_traceback, limit=5)


#----------------------------------------------------------------------------------------#
#
class timer(base_decorator):
    """ A decorator or context-manager for the timer, e.g.:

        class someclass(object):

            @timemory.util.timer()
            def __init__(self):
                self.some_obj = None
        # ...
    """

    #------------------------------------------------------------------------------------#
    #
    def __init__(self, key="", add_args=False, is_class=False, mode="defer"):
        super(timer, self).__init__(key=key, add_args=add_args,
                                    is_class=is_class, mode=mode)
        self._self_obj = None
    #------------------------------------------------------------------------------------#
    #

    def __call__(self, func):
        """
        Decorator
        """
        _file = FILE(3)
        _line = LINE(2)

        @wraps(func)
        def function_wrapper(*args, **kwargs):
            self.parse_wrapped(func, args, kwargs)
            self.determine_signature(
                is_decorator=True, is_context_manager=False)

            _func = func.__name__
            _key = ''
            _args = self.arg_string(args, kwargs)
            if self.signature == context.blank:
                _key = '{}{}'.format(self.key, _args)
            elif self.signature == context.basic:
                _key = '{}/{}/{}'.format(_func, self.key, _args)
            elif self.signature == context.full:
                _key = '{}/{}:{}/{}{}'.format(
                    _func, _file, _line, self.key, _args)
            _key = _key.strip('/')

            t = timer(_key)

            t.start()
            ret = func(*args, **kwargs)
            t.stop()
            t.report()
            return ret

        return function_wrapper
    #------------------------------------------------------------------------------------#
    #

    def __enter__(self, *args, **kwargs):
        """
        Context manager
        """
        _file = FILE(3)
        _line = LINE(2)
        _func = FUNC(2)
        self.determine_signature(is_decorator=False, is_context_manager=True)

        _key = ''
        _args = self.arg_string(args, kwargs)
        if self.signature == context.blank:
            _key = '{}{}'.format(self.key, _args)
        elif self.signature == context.basic:
            _key = '{}/{}/{}'.format(_func, self.key, _args)
        elif self.signature == context.full:
            _key = '{}/{}:{}/{}{}'.format(
                _func, _file, _line, self.key, _args)
        _key = _key.strip('/')

        self._self_obj = timer(_key)
        self._self_obj.start()
    #------------------------------------------------------------------------------------#
    #

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._self_obj.stop()
        self._self_obj.report()

        if exc_type is not None and exc_value is not None and exc_traceback is not None:
            import traceback
            traceback.print_exception(
                exc_type, exc_value, exc_traceback, limit=5)


#----------------------------------------------------------------------------------------#
#
class rss_usage(base_decorator):
    """ A decorator or context-manager for the rss usage, e.g.:

        class someclass(object):

            @timemory.util.rss_usage()
            def __init__(self):
                self.some_obj = None
        # ...
    """

    #------------------------------------------------------------------------------------#
    #
    def __init__(self, key="", add_args=False, is_class=False, mode="defer"):
        super(rss_usage, self).__init__(
            key=key, add_args=add_args, is_class=is_class, mode=mode)
        self._self_obj = None
        self._self_dif = None
    #------------------------------------------------------------------------------------#
    #

    def __call__(self, func):
        """
        Decorator
        """
        _file = FILE(3)
        _line = LINE(2)

        @wraps(func)
        def function_wrapper(*args, **kwargs):
            self.parse_wrapped(func, args, kwargs)
            self.determine_signature(
                is_decorator=True, is_context_manager=False)

            _func = func.__name__
            _key = ''
            _args = self.arg_string(args, kwargs)
            if self.signature == context.blank:
                _key = '{}{}'.format(self.key, _args)
            elif self.signature == context.basic:
                _key = '{}/{}/{}'.format(_func, self.key, _args)
            elif self.signature == context.full:
                _key = '{}/{}:{}/{}{}'.format(
                    _func, _file, _line, self.key, _args)
            _key = _key.strip('/')

            self._self_obj = rss_usage(_key)
            self._self_dif = rss_usage(_key)
            self._self_dif.record()
            # run function
            ret = func(*args, **kwargs)
            # record
            self._self_obj.record()
            self._self_obj -= self._self_dif
            print('{}'.format(self._self_obj))

            return ret

        return function_wrapper
    #------------------------------------------------------------------------------------#
    #

    def __enter__(self, *args, **kwargs):
        """
        Context manager entrance
        """
        _file = FILE(3)
        _line = LINE(2)
        _func = FUNC(2)
        self.determine_signature(is_decorator=False, is_context_manager=True)

        _key = ''
        _args = self.arg_string(args, kwargs)
        if self.signature == context.blank:
            _key = '{}{}'.format(self.key, _args)
        elif self.signature == context.basic:
            _key = '{}/{}/{}'.format(_func, self.key, _args)
        elif self.signature == context.full:
            _key = '{}/{}:{}/{}{}'.format(
                _func, _file, _line, self.key, _args)
        _key = _key.strip('/')

        self._self_obj = rss_usage(_key)
        self._self_dif = rss_usage(_key)
        self._self_dif.record()
    #------------------------------------------------------------------------------------#
    #

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Context manager exit
        """
        self._self_obj.record()
        self._self_obj -= self._self_dif
        print('{}'.format(self._self_obj))

        if exc_type is not None and exc_value is not None and exc_traceback is not None:
            import traceback
            traceback.print_exception(
                exc_type, exc_value, exc_traceback, limit=5)


#----------------------------------------------------------------------------------------#
#
class marker(base_decorator):
    """ A decorator or context-manager for a marker, e.g.:
        @timemory.util.marker(components=("wall_clock", timemory.component.peak_rss))
        def main(n=5):
            for i in range(2):
                fibonacci(n * (i+1))
        # ...
        # output :
        # > [pyc] main(5)@'example.py':10 ...
    """

    #------------------------------------------------------------------------------------#
    #
    @staticmethod
    def get_components(items):
        ret = []
        for x in items:
            if isinstance(x, six.string_types):
                try:
                    ret.append(getattr(component, x))
                except:
                    pass
            else:
                ret.append(x)
        return ret

    #------------------------------------------------------------------------------------#
    #
    def __init__(self, components=[], key="", add_args=False, is_class=False,
                 report_at_exit=False, mode="defer"):
        super(marker, self).__init__(
            key=key, add_args=add_args, is_class=is_class, mode=mode)
        self.components = marker.get_components(components)
        self.report_at_exit = report_at_exit
        self._self_obj = None
    #------------------------------------------------------------------------------------#
    #

    def __call__(self, func):
        """
        Decorator
        """
        _file = FILE(3)
        _line = LINE(2)

        @wraps(func)
        def function_wrapper(*args, **kwargs):
            self.parse_wrapped(func, args, kwargs)
            self.determine_signature(
                is_decorator=True, is_context_manager=False)

            _func = func.__name__
            _key = ''
            _args = self.arg_string(args, kwargs)
            if self.signature == context.blank:
                _key = '{}{}'.format(self.key, _args)
            elif self.signature == context.basic:
                _key = '{}/{}/{}'.format(_func, self.key, _args)
            elif self.signature == context.full:
                _key = '{}/{}:{}/{}{}'.format(
                    _func, _file, _line, self.key, _args)
            _key = _key.strip('/')

            t = component_decorator(self.components, _key)
            ret = func(*args, **kwargs)
            del t
            return ret

        return function_wrapper
    #------------------------------------------------------------------------------------#
    #

    def __enter__(self, *args, **kwargs):
        """
        Context manager
        """
        _file = FILE(3)
        _line = LINE(2)
        _func = FUNC(2)
        self.determine_signature(is_decorator=False, is_context_manager=True)

        _key = ''
        _args = self.arg_string(args, kwargs)
        if self.signature == context.blank:
            _key = '{}{}'.format(self.key, _args)
        elif self.signature == context.basic:
            _key = '{}/{}/{}'.format(_func, self.key, _args)
        elif self.signature == context.full:
            _key = '{}/{}:{}/{}{}'.format(
                _func, _file, _line, self.key, _args)
        _key = _key.strip('/')

        self._self_obj = component_decorator(self.components, _key)
    #------------------------------------------------------------------------------------#
    #

    def __exit__(self, exc_type, exc_value, exc_traceback):
        del self._self_obj

        if exc_type is not None and exc_value is not None and exc_traceback is not None:
            import traceback
            traceback.print_exception(
                exc_type, exc_value, exc_traceback, limit=5)


auto_tuple = marker
