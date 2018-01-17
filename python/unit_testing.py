#!/usr/bin/env python

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
import timemory as timing
import unittest

from os.path import join

timing.enable_signal_detection()

def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

class TimingTest(unittest.TestCase):

    def setUp(self):
        self.outdir = "timemory_test_output"
        timing.util.use_timers = True
        timing.util.serial_report = True


    # Test if the timers are working if not disabled at compilation
    def test_timing(self):
        print ('Testing function: "{}"...'.format(timing.FUNC()))

        timing.util.opts.set_report(join(self.outdir, "timing_report.out"))
        timing.util.opts.set_serial(join(self.outdir, "timing_report.json"))

        tman = timing.timing_manager()

        def time_fibonacci(n):
            atimer = timing.auto_timer('({})@{}'.format(n, timing.FILE(use_dirname=True)))
            key = ('fibonacci(%i)' % n)
            timer = timing.timer(key)
            timer.start()
            fibonacci(n)
            timer.stop()


        tman.clear()
        t = timing.timer("tmanager_test")
        t.start()

        for i in [39, 35, 43, 39]:
            # python is too slow with these values that run in a couple
            # seconds in C/C++
            n = i - 12
            time_fibonacci(n - 2)
            time_fibonacci(n - 1)
            time_fibonacci(n)
            time_fibonacci(n + 1)

        tman.report()

        self.assertEqual(tman.size(), 12)

        for i in range(0, tman.size()):
            t = tman.at(i)
            self.assertFalse(t.real_elapsed() < 0.0)
            self.assertFalse(t.user_elapsed() < 0.0)
        timing.toggle(True)



    # Test the timing on/off toggle functionalities
    def test_toggle(self):
        print ('Testing function: "{}"...'.format(timing.FUNC()))

        tman = timing.timing_manager()
        timing.toggle(True)

        timing.set_max_depth(timing.util.default_max_depth())
        tman.clear()

        timing.toggle(True)
        if True:
            autotimer = timing.auto_timer("on")
            fibonacci(27)
            del autotimer
        self.assertEqual(tman.size(), 1)

        tman.clear()
        timing.toggle(False)
        if True:
            autotimer = timing.auto_timer("off")
            fibonacci(27)
            del autotimer
        self.assertEqual(tman.size(), 0)

        tman.clear()
        timing.toggle(True)
        if True:
            autotimer_on = timing.auto_timer("on")
            timing.toggle(False)
            autotimer_off = timing.auto_timer("off")
            fibonacci(27)
            del autotimer_off
            del autotimer_on
        self.assertEqual(tman.size(), 1)

        timing.util.opts.set_report(join(self.outdir, "timing_toggle.out"))
        timing.util.opts.set_serial(join(self.outdir, "timing_toggle.json"))

        tman.report()


    # Test the timing on/off toggle functionalities
    def test_max_depth(self):
        print ('Testing function: "{}"...'.format(timing.FUNC()))

        tman = timing.timing_manager()
        timing.toggle(True)
        tman.clear()

        def create_timer(n):
            autotimer = timing.auto_timer('{}'.format(n))
            fibonacci(30)
            if n < 8:
                create_timer(n + 1)

        ntimers = 4
        timing.set_max_depth(ntimers)

        create_timer(0)

        timing.util.opts.set_report(join(self.outdir, "timing_depth.out"))
        timing.util.opts.set_serial(join(self.outdir, "timing_depth.json"))
        tman.report()

        self.assertEqual(tman.size(), ntimers)


if __name__ == '__main__':
    unittest.main()
