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

import os
import sys
import copy
from functools import wraps
from collections import deque

from ..libpytimemory.profiler import profiler_function as _profiler_function
from ..libpytimemory.profiler import config as _profiler_config
from ..libpytimemory.profiler import initialize as _profiler_init
from ..libpytimemory.profiler import finalize as _profiler_fini
from ..libpytimemory.profiler import profiler_bundle as _profiler_bundle
from ..libpytimemory import settings
from ..libpytimemory import component_bundle


__all__ = ["profile", "Profiler", "FakeProfiler", "Config"]


#
def _default_functor():
    return True


#
PY3 = sys.version_info[0] == 3
PY35 = PY3 and sys.version_info[1] >= 5

# exec (from https://bitbucket.org/gutworth/six/):
if PY3:
    import builtins

    exec_ = getattr(builtins, "exec")
    del builtins
else:

    def exec_(_code_, _globs_=None, _locs_=None):
        """Execute code in a namespace."""
        if _globs_ is None:
            frame = sys._getframe(1)
            _globs_ = frame.f_globals
            if _locs_ is None:
                _locs_ = frame.f_locals
            del frame
        elif _locs_ is None:
            _locs_ = _globs_
        exec("""exec _code_ in _globs_, _locs_""")


#
Config = _profiler_config


#
class Profiler:
    """
    Provides decorators and context-manager for the timemory profilers
    """

    global _default_functor

    # static variable
    _conditional_functor = _default_functor

    # ------------------------------------------------------------------------------------#
    #
    @staticmethod
    def condition(functor):
        Profiler._conditional_functor = functor

    # ------------------------------------------------------------------------------------#
    #
    @staticmethod
    def is_enabled():
        ret = Profiler._conditional_functor()
        try:
            if ret is True:
                return True
            elif ret is False:
                return False
        except:
            pass
        return False

    # ------------------------------------------------------------------------------------#
    #
    def __init__(
        self, components=[], flat=False, timeline=False, *args, **kwargs
    ):
        """
        Arguments:
            - components [list of strings]  : list of timemory components
            - flat [bool]                   : enable flat profiling
            - timeline [bool]               : enable timeline profiling
        """

        _trace = settings.trace_components
        _profl = settings.profiler_components
        _components = _profl if _trace is None else _trace

        self._original_profiler_function = sys.getprofile()
        self._use = (
            not _profiler_config._is_running and Profiler.is_enabled() is True
        )
        self._flat_profile = settings.flat_profile or flat
        self._timeline_profile = settings.timeline_profile or timeline
        self.components = components + _components.split(",")
        self.components = [v.lower() for v in self.components]
        self.components = list(dict.fromkeys(self.components))
        if len(self.components) == 0:
            self.components += ["wall_clock"]
        if _trace is None:
            settings.trace_components = ",".join(self.components)
        settings.profiler_components = ",".join(self.components)

    # ------------------------------------------------------------------------------------#
    #
    def configure(self):
        _profiler_init()
        _profiler_bundle.configure(
            self.components, self._flat_profile, self._timeline_profile
        )
        self._original_profiler_function = sys.getprofile()

    # ------------------------------------------------------------------------------------#
    #
    def start(self):
        """
        Start the profiler explicitly
        """

        self._use = (
            not _profiler_config._is_running and Profiler.is_enabled() is True
        )
        if self._use and not _profiler_config._is_running:
            self.configure()
            sys.setprofile(_profiler_function)

    # ------------------------------------------------------------------------------------#
    #
    def stop(self):
        """
        Stop the profiler explicitly
        """

        if self._use and _profiler_config._is_running:
            sys.setprofile(self._original_profiler_function)
            _profiler_fini()

    # ------------------------------------------------------------------------------------#
    #
    def __call__(self, func):
        """
        Decorator
        """

        self._use = (
            not _profiler_config._is_running and Profiler.is_enabled() is True
        )
        # print("[__call__:beg]> use: {}, _is_running: {}".format(self._use,
        #      _profiler_config._is_running))

        if self._use and not _profiler_config._is_running:
            self.configure()

        @wraps(func)
        def function_wrapper(*args, **kwargs):
            if self._use and sys.getprofile() != _profiler_function:
                sys.setprofile(_profiler_function)
            return func(*args, **kwargs)

        ret = function_wrapper

        # print("[__call__:end]> use: {}, _is_running: {}".format(self._use,
        #      _profiler_config._is_running))
        self.stop()

        return ret

    # ------------------------------------------------------------------------------------#
    #
    def __enter__(self, *args, **kwargs):
        """
        Context manager start function
        """
        self.start()

    # ------------------------------------------------------------------------------------#
    #
    def __exit__(self, exec_type, exec_value, exec_tb):
        """
        Context manager stop function
        """

        self.stop()
        if (
            exec_type is not None
            and exec_value is not None
            and exec_tb is not None
        ):
            import traceback

            traceback.print_exception(exec_type, exec_value, exec_tb, limit=5)

    # ------------------------------------------------------------------------------------#
    #
    def run(self, cmd):
        """
        Execute and profile a command
        """

        import __main__

        dict = __main__.__dict__
        if isinstance(cmd, str):
            return self.runctx(cmd, dict, dict)
        else:
            return self.runctx(" ".join(cmd), dict, dict)

    # ------------------------------------------------------------------------------------#
    #
    def runctx(self, cmd, globals, locals):
        """
        Profile a context
        """

        if self._use:
            self.configure()
            sys.setprofile(_profiler_function)

        try:
            exec_(cmd, globals, locals)
        finally:
            if self._use:
                sys.setprofile(self._original_profiler_function)
                _profiler_fini()

        return self

    # ------------------------------------------------------------------------------------#
    #
    def runcall(self, func, *args, **kw):
        """
        Profile a single function call.
        """

        ret = None

        if self._use:
            self.configure()
            sys.setprofile(_profiler_function)

        try:
            ret = func(*args, **kw)
        finally:
            if self._use:
                sys.setprofile(self._original_profiler_function)
                _profiler_fini()

        return ret

    # ------------------------------------------------------------------------------------#
    #
    def add_module(self, mod):
        """Add all the functions in a module and its classes."""
        from inspect import isclass, isfunction

        nfuncsadded = 0
        for item in mod.__dict__.values():
            if isclass(item):
                for k, v in item.__dict__.items():
                    if isfunction(v):
                        self.add_function(v)
                        nfuncsadded += 1
            elif isfunction(item):
                self.add_function(item)
                nfuncsadded += 1

        return nfuncsadded


profile = Profiler


class FakeProfiler:
    """
    Provides decorators and context-manager for the timemory profiles which do nothing
    """

    # ------------------------------------------------------------------------------------#
    #
    @staticmethod
    def condition(functor):
        pass

    # ------------------------------------------------------------------------------------#
    #
    @staticmethod
    def is_enabled():
        return False

    # ------------------------------------------------------------------------------------#
    #
    def __init__(self, *args, **kwargs):
        """
        Arguments:
            - components [list of strings]  : list of timemory components
            - flat [bool]                   : enable flat profiling
            - timeline [bool]               : enable timeline profiling
        """
        pass

    # ------------------------------------------------------------------------------------#
    #
    def __call__(self, func):
        """
        Decorator
        """

        @wraps(func)
        def function_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return function_wrapper

    # ------------------------------------------------------------------------------------#
    #
    def __enter__(self, *args, **kwargs):
        """
        Context manager
        """
        pass

    # ------------------------------------------------------------------------------------#
    #
    def __exit__(self, exec_type, exec_value, exec_tb):
        """
        Context manager
        """
        import traceback

        if (
            exec_type is not None
            and exec_value is not None
            and exec_tb is not None
        ):
            traceback.print_exception(exec_type, exec_value, exec_tb, limit=5)
