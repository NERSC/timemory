# MIT License
#
# Copyright (c) 2020, The Regents of the University of California,
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

"""
Imports common utilities
"""

from __future__ import absolute_import

__author__ = "Jonathan Madsen"
__copyright__ = "Copyright 2020, The Regents of the University of California"
__credits__ = ["Jonathan Madsen"]
__license__ = "MIT"
__version__ = "@PROJECT_VERSION@"
__maintainer__ = "Jonathan Madsen"
__email__ = "jrmadsen@lbl.gov"
__status__ = "Development"

import sys
import os
from os.path import dirname
from os.path import basename
from os.path import join

__all__ = [
    "file",
    "func",
    "line",
    "frame",
    "is_generator",
    "is_coroutine",
    "execute_",
    "PY3",
    "PY35",
    "FILE",
    "FUNC",
    "LINE",
    "FRAME",
]

CO_GENERATOR = 0x0020
# Python 2/3 compatibility utils
PY3 = sys.version_info[0] == 3
PY35 = PY3 and sys.version_info[1] >= 5


def frame(back=2):
    """Returns a frame"""
    return sys._getframe(back)


def file(back=2, only_basename=True, use_dirname=False, noquotes=True):
    """
    Returns the file name
    """

    def get_fcode(back):
        fname = "<module>"
        try:
            fname = sys._getframe(back).f_code.co_filename
        except Exception as e:
            print(e)
            fname = "<module>"
        return fname

    result = None
    if only_basename is True:
        if use_dirname is True:
            result = "{}".format(
                join(
                    basename(dirname(get_fcode(back))),
                    basename(get_fcode(back)),
                )
            )
        else:
            result = "{}".format(basename(get_fcode(back)))
    else:
        result = "{}".format(get_fcode(back))

    if noquotes is False:
        result = "'{}'".format(result)

    return result


def func(back=2):
    """
    Returns the function name
    """
    return "{}".format(sys._getframe(back).f_code.co_name)


def line(back=1):
    """
    Returns the line number
    """
    return int(sys._getframe(back).f_lineno)


FRAME = frame
FILE = file
FUNC = func
LINE = line


def is_generator(f):
    """Return True if a function is a generator."""
    isgen = (f.__code__.co_flags & CO_GENERATOR) != 0
    return isgen


# exec (from https://bitbucket.org/gutworth/six/):
if PY3:
    import builtins

    execute_ = getattr(builtins, "exec")
    del builtins
else:

    def execute_(_code_, _globs_=None, _locs_=None):
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


if PY35:
    import inspect

    def is_coroutine(f):
        return inspect.iscoroutinefunction(f)


else:

    def is_coroutine(f):
        return False
