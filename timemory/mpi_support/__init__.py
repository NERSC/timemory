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

from __future__ import absolute_import

import importlib
import sys
import os
import imp

# get the path to this directory
__this_path = os.path.abspath(os.path.dirname(__file__))

def __load_module(module_name, path):
    return imp.load_module(module_name, open(path), path, ('py', 'U', imp.PY_SOURCE))

__self = __load_module('timemory.mpi_support', os.path.join(__this_path, 'mpi_support.py'))
__file_path = __self.__file__
__module_name = __self.__name__

# Python 3.1+
if sys.version_info[0] > 2 and sys.version_info[1] > 0:
    # try to populate __spec__ attribute
    try:
        __spec = importlib.util.spec_from_file_location(__module_name, __file_path)
        __module = importlib.util.module_from_spec(__spec)
        __spec.loader.exec_module(__module)
        sys.modules[__module_name] = __module

    except Exception as e:
        sys.modules[__module_name] = __self
else:
    sys.modules[__module_name] = __self
