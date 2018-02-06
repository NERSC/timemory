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

## @file timemory_test.py
## Unit tests for TiMemory module
##

import sys
import os
import time
import numpy as np
from os.path import join
import unittest as unittest

import timemory
from timemory import options

# ============================================================================ #
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


# ============================================================================ #
class timemory_test(unittest.TestCase):

    # ------------------------------------------------------------------------ #
    def __init__(self, *args, **kwargs):
        super(timemory_test, self).__init__(*args, **kwargs)

    # ------------------------------------------------------------------------ #
    def setUp(self):
        timemory.options.output_dir = "test_output"
        timemory.options.use_timers = True
        timemory.options.serial_report = True
        self.timing_manager = timemory.timing_manager()

    # ------------------------------------------------------------------------ #
    # Test if the timers are working if not disabled at compilation
    def test_timing(self):
        print ('\n\n--> Testing function: "{}"...\n\n'.format(timemory.FUNC()))

        options.set_report("timing_report.out")
        options.set_serial("timing_report.json")

        def time_fibonacci(n):
            atimer = timemory.auto_timer('({})@{}'.format(n, timemory.FILE(use_dirname=True)))
            key = ('fibonacci(%i)' % n)
            timer = timemory.timer(key)
            timer.start()
            fibonacci(n)
            timer.stop()


        self.timing_manager.clear()
        t = timemory.timer("tmanager_test")
        t.start()

        for i in [39, 35, 43, 39]:
            # python is too slow with these values that run in a couple
            # seconds in C/C++
            n = i - 12
            time_fibonacci(n - 2)
            time_fibonacci(n - 1)
            time_fibonacci(n)
            time_fibonacci(n + 1)

        self.timing_manager.merge()
        self.timing_manager.report()

        self.assertEqual(self.timing_manager.size(), 12)

        for i in range(0, self.timing_manager.size()):
            t = self.timing_manager.at(i)
            self.assertFalse(t.real_elapsed() < 0.0)
            self.assertFalse(t.user_elapsed() < 0.0)

        timemory.toggle(True)


    # ------------------------------------------------------------------------ #
    # Test the timing on/off toggle functionalities
    def test_toggle(self):
        print ('\n\n--> Testing function: "{}"...\n\n'.format(timemory.FUNC()))

        timemory.toggle(True)
        timemory.set_max_depth(timemory.options.default_max_depth())
        self.timing_manager.clear()

        timemory.toggle(True)
        if True:
            autotimer = timemory.auto_timer("on")
            fibonacci(27)
            del autotimer
        self.assertEqual(self.timing_manager.size(), 1)

        self.timing_manager.clear()
        timemory.toggle(False)
        if True:
            autotimer = timemory.auto_timer("off")
            fibonacci(27)
            del autotimer
        self.assertEqual(self.timing_manager.size(), 0)

        self.timing_manager.clear()
        timemory.toggle(True)
        if True:
            autotimer_on = timemory.auto_timer("on")
            timemory.toggle(False)
            autotimer_off = timemory.auto_timer("off")
            fibonacci(27)
            del autotimer_off
            del autotimer_on
        self.assertEqual(self.timing_manager.size(), 1)

        timemory.options.set_report("timing_toggle.out")
        timemory.options.set_serial("timing_toggle.json")

        self.timing_manager.report()


    # ------------------------------------------------------------------------ #
    # Test the timing on/off toggle functionalities
    def test_max_depth(self):
        print ('\n\n--> Testing function: "{}"...\n\n'.format(timemory.FUNC()))

        timemory.toggle(True)
        self.timing_manager.clear()

        def create_timer(n):
            autotimer = timemory.auto_timer('{}'.format(n))
            fibonacci(30)
            if n < 8:
                create_timer(n + 1)

        ntimers = 4
        timemory.set_max_depth(ntimers)

        create_timer(0)

        timemory.options.set_report("timing_depth.out")
        timemory.options.set_serial("timing_depth.json")
        self.timing_manager.report()

        self.assertEqual(self.timing_manager.size(), ntimers)


    # ------------------------------------------------------------------------ #
    # Test the timing on/off toggle functionalities
    def test_pointer(self):
        print ('\n\n--> Testing function: "{}"...\n\n'.format(timemory.FUNC()))

        nval = 4

        def set_pointer_max(nmax):
            self.timing_manager.set_max_depth(4)
            return self.timing_manager.get_max_depth()

        def get_pointer_max():
            return timemory.timing_manager().get_max_depth()

        ndef = get_pointer_max()
        nnew = set_pointer_max(nval)
        nchk = get_pointer_max()

        self.assertEqual(nval, nchk)

        set_pointer_max(ndef)

        self.assertEqual(ndef, get_pointer_max())


    # ------------------------------------------------------------------------ #
    # Test decorator
    def test_decorator(self):
        print ('\n\n--> Testing function: "{}"...\n\n'.format(timemory.FUNC()))

        import time

        timemory.clear()

        @timemory.util.auto_timer()
        def test_func_glob():
            time.sleep(1)

            @timemory.util.auto_timer()
            def test_func_1():
                time.sleep(1)

            @timemory.util.auto_timer()
            def test_func_2(n):
                test_func_1()
                time.sleep(n)

            test_func_1()
            test_func_2(2)

        test_func_glob()

        timemory.report()

        self.assertEqual(timemory.size(), 4)

        @timemory.util.timer()
        def test_func_timer():
            time.sleep(1)

            @timemory.util.rss_usage()
            def test_func_rss():
                ret = np.ones(shape=[3000, 3000], dtype=np.float128)
                return None

            print('')
            ret = test_func_rss()
            print('')
            #print('ret: {}'.format(ret), flush=True)
            time.sleep(1)
            return None

        test_func_timer()


# ---------------------------------------------------------------------------- #
def run_test():

    _test = timemory_test()
    _test.setUp()
    _test.test_decorator()
    _test.setUp()
    _test.test_max_depth()
    _test.setUp()
    _test.test_pointer()
    _test.setUp()
    _test.test_timing()
    _test.setUp()
    _test.test_toggle()


# ---------------------------------------------------------------------------- #
if __name__ == '__main__':
    unittest.main()
