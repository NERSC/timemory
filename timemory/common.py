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

__all__ = ['FILE', 'FUNC', 'LINE']


def FILE(back=2, only_basename=True, use_dirname=False,
         noquotes=True):
    """
    Returns the file name
    """
    def get_fcode(back):
        fname = '<module>'
        try:
            fname = sys._getframe(back).f_code.co_filename
        except Exception as e:
            print(e)
            fname = '<module>'
        return fname

    result = None
    if only_basename is True:
        if use_dirname is True:
            result = ("{}".format(join(basename(dirname(get_fcode(back))),
                                       basename(get_fcode(back)))))
        else:
            result = ("{}".format(basename(get_fcode(back))))
    else:
        result = ("{}".format(get_fcode(back)))

    if noquotes is False:
        result = ("'{}'".format(result))

    return result


def FUNC(back=2):
    """
    Returns the function name
    """
    return ("{}".format(sys._getframe(back).f_code.co_name))


def LINE(back=1):
    """
    Returns the line number
    """
    return int(sys._getframe(back).f_lineno)
