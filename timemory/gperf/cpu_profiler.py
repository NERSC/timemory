#!@PYTHON_EXECUTABLE@
# MIT License
#
# Copyright (c) 2019, The Regents of the University of California,
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
from . import general


def execute(cmd, prefix="cpu.prof", freq=250, malloc_stats=0, realtime=1, preload=False,
            args=["--no_strip_temp", "--functions"]):

    os.environ["CPUPROFILE_FREQUENCY"] = "{}".format(freq)
    os.environ["MALLOCSTATS"] = "{}".format(malloc_stats)
    os.environ["CPUPROFILE_REALTIME"] = "{}".format(realtime)

    if preload:
        _libpath = general.find_library_path("libprofiler")
        if _libpath is not None:
            raise RuntimeError("Preload failed. Cannot find libprofiler")
        general.add_preload(_libpath)

    exe = os.path.basename(cmd[0])
    prefix += ".{}".format(exe)
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    N = 0
    fprefix = os.path.join(prefix, "gperf.")
    fname = fprefix + "{}".format(N)
    while os.path.exists(fname):
        N += 1
        fname = fprefix + "{}".format(N)

    os.environ["CPUPROFILE"] = "{}".format(fname)

    general.execute(cmd, "{}.log".format(fname))

    _libs = general.get_linked_libraries(exe)
    _liblist = []
    if _libs is not None:
        for _lib in _libs:
            _liblist += ["--add_lib={}".format(_lib)]

    general.execute(["google-pprof", "--text"] + _liblist +
                    args + [exe, fname], "{}.txt".format(fname))
    general.execute(["google-pprof", "--text", "--cum"] + _liblist +
                    args + [exe, fname], "{}.cum.txt".format(fname))
    general.execute(["google-pprof", "--dot"] + _liblist +
                    args + [exe, fname], "{}.dot".format(fname))
    general.execute(["google-pprof", "--callgrind"] + _liblist +
                    args + [exe, fname], "{}.callgrind".format(fname))
