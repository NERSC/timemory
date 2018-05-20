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

## @file mpi_support.py
## MPI information for TiMemory module
##

from __future__ import absolute_import
import os


#------------------------------------------------------------------------------#
def _path_to_file(pfile):
    return '{}/{}'.format(os.path.dirname(__file__), pfile)


#------------------------------------------------------------------------------#
def mpi_exe_info():
    print (open(_path_to_file('mpi_exe_info.txt'), 'r').read())


#------------------------------------------------------------------------------#
def mpi_c_info():
    print (open(_path_to_file('mpi_c_info.txt'), 'r').read())


#------------------------------------------------------------------------------#
def mpi_cxx_info():
    print (open(_path_to_file('mpi_cxx_info.txt'), 'r').read())


#------------------------------------------------------------------------------#
def is_supported():
    import timemory
    return timemory.has_mpi_support()

