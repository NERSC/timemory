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

## @file timemory_test.py
## Unit tests for TiMemory module
##

import sys
import os
import time
import numpy as np
from os.path import join
import unittest as unittest
import traceback
import argparse

import timemory
import timemory.options as options
import timemory.plotting as plotting

# decorators
from timemory.util import auto_timer
from timemory.util import rss_usage
from timemory.util import timer

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
        self.output_dir = "test_output"
        timemory.options.output_dir = self.output_dir
        timemory.options.use_timers = True
        timemory.options.serial_report = True
        self.manager = timemory.manager()


    # ------------------------------------------------------------------------ #
    # Test RSS usage validity
    def test_1_rss_validity(self):

        import numpy as np

        print ('\n\n--> Testing function: "{}"...\n\n'.format(timemory.FUNC()))

        rss_init = timemory.rss_usage(record=True)
        rss_post = timemory.rss_usage(record=False)

        print('\t(A) RSS post: {}'.format(rss_post))
        print('\t(A) RSS init: {}'.format(rss_init))

        # should be 8 MB
        nsize = 1048576
        arr1 = np.ones(shape=[nsize], dtype=np.uint64, order='C')

        #for i in range(nsize):
        #    arr1[i] = i

        rss_post.record()

        print('\t(B) RSS post: {}'.format(rss_post))
        print('\t(B) RSS init: {}'.format(rss_init))

        rss_post -= rss_init

        print('\t(C) RSS post: {}'.format(rss_post))
        print('\t(C) RSS init: {}'.format(rss_init))

        # in kB
        rss_corr = 8192
        # real memory in kilobytes
        rss_real = rss_post.current(timemory.units.kilobyte)

        # compute diff
        rss_diff = rss_real - rss_corr
        print('\tRSS real:  {} kB'.format(rss_real))
        print('\tRSS ideal: {} kB'.format(rss_corr))
        print('\tRSS diff:  {} kB'.format(rss_diff))

        # allow some variability
        self.assertTrue(abs(rss_diff) < 300)


    # ------------------------------------------------------------------------ #
    # Test if the timers are working if not disabled at compilation
    def test_2_timing(self):
        print ('\n\n--> Testing function: "{}"...\n\n'.format(timemory.FUNC()))
        self.manager.clear()

        freport = options.set_report("timing_report.out")
        fserial = options.set_serial("timing_report.json")

        def time_fibonacci(n):
            atimer = timemory.auto_timer('({})@{}'.format(n, timemory.FILE(use_dirname=True)))
            key = ('fibonacci(%i)' % n)
            timer = timemory.timer(key)
            timer.start()
            fibonacci(n)
            timer.stop()


        self.manager.clear()
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

        self.manager.merge()
        self.manager.report()
        plotting.plot(files=[fserial], output_dir=self.output_dir)

        self.assertEqual(self.manager.size(), 13)

        for i in range(0, self.manager.size()):
            _t = self.manager.at(i)
            self.assertFalse(_t.real_elapsed() < 0.0)
            self.assertFalse(_t.user_elapsed() < 0.0)

        timemory.toggle(True)
        t.stop()
        print('{}'.format(t))


    # ------------------------------------------------------------------------ #
    # Test decorator
    def test_3_decorator(self):
        print ('\n\n--> Testing function: "{}"...\n\n'.format(timemory.FUNC()))

        timemory.toggle(True)
        self.manager.clear()

        @auto_timer()
        def test_func_glob():
            time.sleep(1)

            @auto_timer()
            def test_func_1():
                ret = np.ones(shape=[2500, 2500], dtype=np.float64)
                time.sleep(1)

            @auto_timer()
            def test_func_2(n):
                test_func_1()
                time.sleep(n)

            test_func_1()
            test_func_2(2)

        test_func_glob()

        freport = timemory.options.set_report("timing_decorator.out")
        fserial = timemory.options.set_serial("timing_decorator.json")
        self.manager.report(ign_cutoff=True)
        plotting.plot(files=[fserial], output_dir=self.output_dir)

        self.assertEqual(timemory.size(), 5)

        @timer()
        def test_func_timer():
            time.sleep(1)

            @rss_usage()
            def test_func_rss():
                ret = np.ones(shape=[5000, 5000], dtype=np.float64)
                return None

            print('')
            ret = test_func_rss()
            print('')
            time.sleep(1)
            return None

        test_func_timer()


    # ------------------------------------------------------------------------ #
    # Test the timing on/off toggle functionalities
    def test_4_max_depth(self):
        print ('\n\n--> Testing function: "{}"...\n\n'.format(timemory.FUNC()))

        timemory.toggle(True)
        self.manager.clear()

        def create_timer(n):
            autotimer = timemory.auto_timer('{}'.format(n))
            fibonacci(30)
            if n < 8:
                create_timer(n + 1)

        ntimers = 4
        timemory.set_max_depth(ntimers)

        create_timer(0)

        freport = timemory.options.set_report("timing_depth.out")
        fserial = timemory.options.set_serial("timing_depth.json")
        self.manager.report()
        plotting.plot(files=[fserial], output_dir=self.output_dir)

        self.assertEqual(self.manager.size(), ntimers)


    # ------------------------------------------------------------------------ #
    # Test the persistancy of the manager pointer
    def test_5_pointer(self):
        print ('\n\n--> Testing function: "{}"...\n\n'.format(timemory.FUNC()))

        nval = 4

        def set_pointer_max(nmax):
            self.manager.set_max_depth(4)
            return self.manager.get_max_depth()

        def get_pointer_max():
            return timemory.manager().get_max_depth()

        ndef = get_pointer_max()
        nnew = set_pointer_max(nval)
        nchk = get_pointer_max()

        self.assertEqual(nval, nchk)

        set_pointer_max(ndef)

        self.assertEqual(ndef, get_pointer_max())


    # ------------------------------------------------------------------------ #
    # Test the timing on/off toggle functionalities
    def test_6_toggle(self):
        print ('\n\n--> Testing function: "{}"...\n\n'.format(timemory.FUNC()))

        timemory.toggle(True)
        timemory.set_max_depth(timemory.options.default_max_depth())
        self.manager.clear()

        timemory.toggle(True)
        if True:
            autotimer = timemory.auto_timer("on")
            fibonacci(27)
            del autotimer
        self.assertEqual(self.manager.size(), 2)

        timemory.toggle(False)
        if True:
            autotimer = timemory.auto_timer("off")
            fibonacci(27)
            del autotimer
        self.assertEqual(self.manager.size(), 2)

        timemory.toggle(True)
        if True:
            autotimer_on = timemory.auto_timer("on")
            timemory.toggle(False)
            autotimer_off = timemory.auto_timer("off")
            fibonacci(27)
            del autotimer_off
            del autotimer_on
        self.assertEqual(self.manager.size(), 3)

        freport = timemory.options.set_report("timing_toggle.out")
        fserial = timemory.options.set_serial("timing_toggle.json")
        self.manager.report(ign_cutoff=True)
        plotting.plot(files=[fserial], output_dir=self.output_dir)


    # ------------------------------------------------------------------------ #
    # Test context manager
    def test_7_context_manager(self):
        print ('\n\n--> Testing function: "{}"...\n\n'.format(timemory.FUNC()))

        timemory.toggle(True)
        self.manager.clear()

        #----------------------------------------------------------------------#
        # timer test
        with timemory.util.timer():
            time.sleep(1)

            ret = np.ones(shape=[500, 500], dtype=np.float64)
            for i in [ 2.0, 3.5, 8.7 ]:
                n = i * np.ones(shape=[500, 500], dtype=np.float64)
                ret += n
                del n

        #----------------------------------------------------------------------#
        # timer test with args
        with timemory.util.timer('{}({})'.format(timemory.FUNC(), ['test', 'context'])):
            time.sleep(1)

            ret = np.ones(shape=[500, 500], dtype=np.float64)
            for i in [ 2.0, 3.5, 8.7 ]:
                n = i * np.ones(shape=[500, 500], dtype=np.float64)
                ret += n
                del n


        #----------------------------------------------------------------------#
        # auto timer test
        with timemory.util.auto_timer(report_at_exit=True):
            time.sleep(1)

            ret = np.ones(shape=[500, 500], dtype=np.float64)
            for i in [ 2.0, 3.5, 8.7 ]:
                n = i * np.ones(shape=[500, 500], dtype=np.float64)
                ret += n
                del n

        #----------------------------------------------------------------------#
        # failure test
        def fail_func():
            return [0, 1, 2, 3]

        # timer exit test
        try:
            with timemory.util.timer():
                a, b, c = fail_func()
        except Exception as e:
            pass

        # auto timer exit test
        try:
            with timemory.util.auto_timer():
                a, b, c = fail_func()
        except Exception as e:
            pass

        # rss exit test
        try:
            with timemory.util.rss_usage():
                a, b, c = fail_func()
        except Exception as e:
            pass

        #----------------------------------------------------------------------#
        # auto timer test with args
        @timemory.util.auto_timer()
        def auto_timer_with_args(nc=2):
            # construct context-manager with args
            with timemory.util.auto_timer('{}({})'.format(timemory.FUNC(), ['test', 'context'])):
                time.sleep(nc)

                with timemory.util.auto_timer('{}({})'.format(timemory.FUNC(), ['test', 'context']), report_at_exit=True):
                    ret = np.ones(shape=[500, 500], dtype=np.float64)
                    for i in [ 2.0, 3.5, 8.7 ]:
                        n = i * np.ones(shape=[500, 500], dtype=np.float64)
                        ret += n
                        del n
                # recursive
                if nc > 1:
                    auto_timer_with_args(nc=1)

        auto_timer_with_args()

        print('\n\n{}\n\n'.format(self.manager))

        #----------------------------------------------------------------------#
        # rss test
        with timemory.util.rss_usage():
            time.sleep(1)

            ret = np.ones(shape=[500, 500], dtype=np.float64)
            for i in [ 2.0, 3.5, 8.7 ]:
                n = i * np.ones(shape=[500, 500], dtype=np.float64)
                ret += n
                del n

        #----------------------------------------------------------------------#
        # rss test with args
        with timemory.util.rss_usage('{}({})'.format(timemory.FUNC(), ['test', 'context'])):
            time.sleep(1)

            ret = np.ones(shape=[500, 500], dtype=np.float64)
            for i in [ 2.0, 3.5, 8.7 ]:
                n = i * np.ones(shape=[500, 500], dtype=np.float64)
                ret += n
                del n

        freport = timemory.options.set_report("timing_context_manager.out")
        fserial = timemory.options.set_serial("timing_context_manager.json")
        self.manager.report(ign_cutoff=True)
        plotting.plot(files=[fserial], output_dir=self.output_dir)


    # ------------------------------------------------------------------------ #
    # Test if the timers are working if not disabled at compilation
    def test_8_format(self):

        print ('\n\n--> Testing function: "{}"...\n\n'.format(timemory.FUNC()))
        self.manager.clear()

        t1 = timemory.timer("format_test")
        t2 = timemory.timer("format_test")
        u1 = timemory.rss_usage("format_test")
        u2 = timemory.rss_usage("format_test")

        # Python 2.7 doesn't like timemory.format.{timer,rss} without instance
        if sys.version_info[0] > 2:
            timer_rss_fmt = timemory.format.rss()
            timer_rss_fmt.set_format("%C, %M, %c, %m")
            timer_rss_fmt.set_precision(0)
            timer_rss_fmt.set_unit(timemory.units.kilobyte)

            default_timer_fmt = timemory.format.timer.get_default()
            timemory.format.timer.set_default_format("[%T - %A] : %w, %u, %s, %t, %p%, x%l, %R")
            timemory.format.timer.set_default_rss_format("%C, %M, %c, %m")
            timemory.format.timer.set_default_rss_format(timer_rss_fmt)
            timemory.format.timer.set_default_unit(timemory.units.msec)
            timemory.format.timer.set_default_precision(1)

            default_rss_fmt = timemory.format.rss.get_default()
            timemory.format.rss.set_default_format("[ c, p %A ] : %C, %M")
            timemory.format.rss.set_default_unit(timemory.units.kilobyte)
            timemory.format.rss.set_default_precision(3)

            t1 = timemory.timer("format_test")
            t2 = timemory.timer("format_test")
            u1 = timemory.rss_usage("format_test")
            u2 = timemory.rss_usage("format_test")

            t2.set_format(t2.get_format().copy_from(default_timer_fmt))
            u2.set_format(u2.get_format().copy_from(default_rss_fmt))

        else:
            timer_rss_fmt = timemory.format.rss()
            timer_rss_fmt.set_format("%C, %M, %c, %m")
            timer_rss_fmt.set_precision(0)
            timer_rss_fmt.set_unit(timemory.units.kilobyte)

            timer_format = timemory.format.timer()
            default_timer_fmt = timer_format.get_default()
            timer_format.set_default_format("[%T - %A] : %w, %u, %s, %t, %p%, x%l, %R")
            timer_format.set_default_rss_format("%C, %M, %c, %m")
            timer_format.set_default_rss_format(timer_rss_fmt)
            timer_format.set_default_unit(timemory.units.msec)
            timer_format.set_default_precision(1)

            rss_format = timemory.format.rss()
            default_rss_fmt = rss_format.get_default()
            rss_format.set_default_format("[ c, p %A ] : %C, %M")
            rss_format.set_default_unit(timemory.units.kilobyte)
            rss_format.set_default_precision(3)

            t1 = timemory.timer("format_test")
            t2 = timemory.timer("format_test")
            u1 = timemory.rss_usage("format_test")
            u2 = timemory.rss_usage("format_test")

            t2.set_format(t2.get_format().copy_from(default_timer_fmt))
            u2.set_format(u2.get_format().copy_from(default_rss_fmt))

        freport = options.set_report("timing_format.out")
        fserial = options.set_serial("timing_format.json")

        def time_fibonacci(n):
            atimer = timemory.auto_timer('({})@{}'.format(n, timemory.FILE(use_dirname=True)))
            key = ('fibonacci(%i)' % n)
            timer = timemory.timer(key)
            timer.start()
            fibonacci(n)
            timer.stop()

        self.manager.clear()

        t1.start()
        t2.start()

        for i in [39, 35, 39]:
            # python is too slow with these values that run in a couple
            # seconds in C/C++
            n = i - 12
            time_fibonacci(n - 2)
            time_fibonacci(n - 1)
            time_fibonacci(n)
            time_fibonacci(n + 1)

        self.manager.report()
        plotting.plot(files=[fserial], output_dir=self.output_dir)

        self.assertEqual(self.manager.size(), 9)

        for i in range(0, self.manager.size()):
            _t = self.manager.at(i)
            self.assertFalse(_t.real_elapsed() < 0.0)
            self.assertFalse(_t.user_elapsed() < 0.0)

        timemory.toggle(True)
        self.manager.clear()

        u1.record()
        u2.record()

        print('\n')
        print('[memory] {}'.format(u1))
        print('[memory] {}'.format(u2))

        t1.stop()
        t2.stop()

        print('\n')
        print('[timing] {}'.format(t1))
        print('[timing] {}'.format(t2))

        print('\n')


# ---------------------------------------------------------------------------- #
def run_test():
    try:
        _test = timemory_test()
        _test.setUp()
        _test.test_1_rss_validity()
        _test.setUp()
        _test.test_2_timing()
        _test.setUp()
        _test.test_3_decorator()
        _test.setUp()
        _test.test_4_max_depth()
        _test.setUp()
        _test.test_5_pointer()
        _test.setUp()
        _test.test_6_toggle()
        _test.setUp()
        _test.test_7_context_manager()
        _test.setUp()
        _test.test_8_format()
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=5)
        print ('Exception - {}'.format(e))
        raise


# ---------------------------------------------------------------------------- #
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-explicit", action='store_true', default=False)
    # this variant will remove TiMemory arguments from sys.argv
    args = options.add_args_and_parse_known(parser)

    try:
        if args.run_explicit:
            run_test()
        else:
            loader = unittest.defaultTestLoader.sortTestMethodsUsing = None
            unittest.main(verbosity=5, buffer=False)
        print('"{}" testing finished'.format(__file__))
    except:
        raise
    finally:
        manager = timemory.manager()
        print("{}".format(timemory.get_missing_report()))
        if options.ctest_notes:
            f = manager.write_ctest_notes(directory="test_output/timemory_test")
            print('"{}" wrote CTest notes file : {}'.format(__file__, f))
        print('# of plotted files: {}'.format(len(plotting.plotted_files)))
        for p in plotting.plotted_files:
            n = p[0]
            f = p[1]
            plotting.echo_dart_tag(n, f)
