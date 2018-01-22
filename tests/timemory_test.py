#!/usr/bin/env python

## @package timemory_test.py
## @file unit_testing.py
## Unit tests for TiMemory module
##

# MIT License
#
# Copyright (c) 2018 Jonathan R. Madsen
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

import sys
import os
import time
from os.path import join

import unittest
import timemory

timemory.enable_signal_detection()

def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

class timemory_test(unittest.TestCase):

    def setUp(self):
        self.outdir = "test_output"
        timemory.util.use_timers = True
        timemory.util.serial_report = True


    # Test if the timers are working if not disabled at compilation
    def test_timing(self):
        print ('Testing function: "{}"...'.format(timemory.FUNC()))

        timemory.util.opts.set_report(join(self.outdir, "timing_report.out"))
        timemory.util.opts.set_serial(join(self.outdir, "timing_report.json"))

        tman = timemory.timing_manager()

        def time_fibonacci(n):
            atimer = timemory.auto_timer('({})@{}'.format(n, timemory.FILE(use_dirname=True)))
            key = ('fibonacci(%i)' % n)
            timer = timemory.timer(key)
            timer.start()
            fibonacci(n)
            timer.stop()


        tman.clear()
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

        tman.merge()
        tman.report()

        self.assertEqual(tman.size(), 12)

        for i in range(0, tman.size()):
            t = tman.at(i)
            self.assertFalse(t.real_elapsed() < 0.0)
            self.assertFalse(t.user_elapsed() < 0.0)
        timemory.toggle(True)



    # Test the timing on/off toggle functionalities
    def test_toggle(self):
        print ('Testing function: "{}"...'.format(timemory.FUNC()))

        tman = timemory.timing_manager()
        timemory.toggle(True)

        timemory.set_max_depth(timemory.util.default_max_depth())
        tman.clear()

        timemory.toggle(True)
        if True:
            autotimer = timemory.auto_timer("on")
            fibonacci(27)
            del autotimer
        self.assertEqual(tman.size(), 1)

        tman.clear()
        timemory.toggle(False)
        if True:
            autotimer = timemory.auto_timer("off")
            fibonacci(27)
            del autotimer
        self.assertEqual(tman.size(), 0)

        tman.clear()
        timemory.toggle(True)
        if True:
            autotimer_on = timemory.auto_timer("on")
            timemory.toggle(False)
            autotimer_off = timemory.auto_timer("off")
            fibonacci(27)
            del autotimer_off
            del autotimer_on
        self.assertEqual(tman.size(), 1)

        timemory.util.opts.set_report(join(self.outdir, "timing_toggle.out"))
        timemory.util.opts.set_serial(join(self.outdir, "timing_toggle.json"))

        tman.report()


    # Test the timing on/off toggle functionalities
    def test_max_depth(self):
        print ('Testing function: "{}"...'.format(timemory.FUNC()))

        tman = timemory.timing_manager()
        timemory.toggle(True)
        tman.clear()

        def create_timer(n):
            autotimer = timemory.auto_timer('{}'.format(n))
            fibonacci(30)
            if n < 8:
                create_timer(n + 1)

        ntimers = 4
        timemory.set_max_depth(ntimers)

        create_timer(0)

        timemory.util.opts.set_report(join(self.outdir, "timing_depth.out"))
        timemory.util.opts.set_serial(join(self.outdir, "timing_depth.json"))
        tman.report()

        self.assertEqual(tman.size(), ntimers)

    # Test the timing on/off toggle functionalities
    def test_pointer(self):
        print ('Testing function: "{}"...'.format(timemory.FUNC()))

        nval = 4

        def set_pointer_max(nmax):
            tman = timemory.timing_manager()
            tman.set_max_depth(4)
            return tman.get_max_depth()

        def get_pointer_max():
            return timemory.timing_manager().get_max_depth()

        ndef = get_pointer_max()
        nnew = set_pointer_max(nval)
        nchk = get_pointer_max()

        self.assertEqual(nval, nchk)

        set_pointer_max(ndef)

        self.assertEqual(ndef, get_pointer_max())


if __name__ == '__main__':
    unittest.main()
