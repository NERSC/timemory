#!@PYTHON_EXECUTABLE@

import os
import time
import unittest
import threading
import inspect
import timemory as tim
from timemory import components as comp

# --------------------------- tolerance variables -------------------------------------- #
# deltas for assert
deltatimer = 0.015
deltautil = 2.5

# --------------------------- helper functions ----------------------------------------- #

# check availability of a component
def check_available(component):
    return inspect.isclass(component)

# compute fibonacci
def fibonacci(n):
    return n if n < 2 else (fibonacci(n-1) + fibonacci(n-2))

# cpu utilization for n milliseconds
def consume(n):
    # a mutex held by one lock
    mutex = threading.Lock()
    # acquire lock
    mutex.acquire(True)
    # get current time in nsec
    now = time.time_ns()
    # try until time point
    while(time.time_ns() < (now + (n * 1e6))):
        mutex.acquire(False)

# -------------------------- Timing Tests set ---------------------------------------- #
# Timing tests class
class TiMemoryTimingTests(unittest.TestCase):
    # setup class: timemory settings
    @classmethod
    def setUpClass(self):
        tim.settings.timing_units     = "sec"
        tim.settings.timing_precision = 9
        tim.settings.json_output = True
        tim.timemory_init() # need args? timemory is inited as soon as you import timemory
        tim.settings.dart_output = True
        tim.settings.dart_count = True
        tim.settings.dart_count = 1
        tim.settings.banner = False

    # tear down class: timemory_finalize
    @classmethod
    def tearDownClass(self):
        #tim.timemory_finalize()
        # tim.dmp.finalize()
        pass

# -------------------------------------------------------------------------------------- #
    # test wall timer
    def test_wall_timer(self):
        """
        wall timer
        """
        if not check_available(comp.monotonic_clock):
            raise unittest.SkipTest('[{}] not available'.format('wall_clock'))

        obj = comp.wall_clock("wall_test")
        #obj.push()
        obj.start()
        time.sleep(1)
        obj.stop()
        #obj.pop()

        print("\n[{}]> result: {}".format(self.shortDescription(), obj.get_labeled()))

        # check for near equal
        self.assertAlmostEqual(1.0, obj.get()[0], delta=deltatimer)

# -------------------------------------------------------------------------------------- #
    # test monotonic timer
    def test_monotonic_timer(self):
        """
        monotonic timer
        """
        if not check_available(comp.monotonic_clock):
            raise unittest.SkipTest('[{}] not available'.format('monotonic_clock')) 

        obj = comp.monotonic_clock("mono_test")
        #obj.push()
        obj.start()
        time.sleep(1)
        obj.stop()
        #obj.pop()

        print("\n[{}]> result: {}".format(self.shortDescription(), obj.get_labeled()))

        # check for near equal
        self.assertAlmostEqual(1.0, obj.get()[0], delta=deltatimer)

# -------------------------------------------------------------------------------------- #
    # test monotonic timer raw
    def test_monotonic_raw_timer(self):
        """
        monotonic raw timer
        """
        if not check_available(comp.monotonic_raw_clock):
            raise unittest.SkipTest('[{}] not available'.format('monotonic_raw_clock')) 

        obj = comp.monotonic_raw_clock("mono_raw_test")
        #obj.push()
        obj.start()
        time.sleep(1)
        obj.stop()
        #obj.pop()

        print("\n[{}]> result: {}".format(self.shortDescription(), obj.get_labeled()))

        # check for near equal
        self.assertAlmostEqual(1.0, obj.get()[0], delta=deltatimer)

# -------------------------------------------------------------------------------------- #
    # test system timer
    def test_system_timer(self):
        """
        system timer
        """
        if not check_available(comp.sys_clock):
            raise unittest.SkipTest('[{}] not available'.format('sys_clock')) 

        obj = comp.sys_clock("sys_test")
        #obj.push()
        obj.start()
        time.sleep(1)
        obj.stop()
        #obj.pop()

        print("\n[{}]> result: {}".format(self.shortDescription(), obj.get_labeled()))

        # check for near equal
        self.assertAlmostEqual(0.0, obj.get()[0], delta=deltatimer)

# -------------------------------------------------------------------------------------- #
    # test user timer
    def test_user_timer(self):
        """
        user timer
        """
        if not check_available(comp.user_clock):
            raise unittest.SkipTest('[{}] not available'.format('user_clock')) 

        obj = comp.user_clock("user_test")
        #obj.push()
        obj.start()
        time.sleep(1)
        obj.stop()
        #obj.pop()

        print("\n[{}]> result: {}".format(self.shortDescription(), obj.get_labeled()))

        # check for near equal
        self.assertAlmostEqual(0.0, obj.get()[0], delta=deltatimer)

# -------------------------------------------------------------------------------------- #
    # test cpu timer
    def test_cpu_timer(self):
        """
        cpu timer
        """
        if not check_available(comp.cpu_clock):
            raise unittest.SkipTest('[{}] not available'.format('cpu_clock')) 

        obj = comp.cpu_clock("cpu_test")
        #obj.push()
        obj.start()
        time.sleep(1)
        obj.stop()
        #obj.pop()

        print("\n[{}]> result: {}".format(self.shortDescription(), obj.get_labeled()))

        # check for near equal
        self.assertAlmostEqual(0.0, obj.get()[0], delta=deltatimer)

# -------------------------------------------------------------------------------------- #
    # test cpu utilization
    def test_cpu_util(self):
        """
        cpu utilization
        """
        if not check_available(comp.cpu_util):
            raise unittest.SkipTest('[{}] not available'.format('cpu_util')) 

        obj = comp.cpu_util("cpu_util")
        #obj.push()
        obj.start()
        # work for 750ms
        consume(750)
        # sleep for 250ms
        time.sleep(0.25)
        obj.stop()
        #obj.pop()

        print("\n[{}]> result: {}".format(self.shortDescription(), obj.get_labeled()))

        # check for near equal
        self.assertAlmostEqual(75.0, obj.get()[0], delta=deltautil)

# -------------------------------------------------------------------------------------- #
    # test thread cpu clock
    def test_thread_cpu_timer(self):
        """
        thread cpu timer 
        """
        if not check_available(comp.thread_cpu_clock):
            raise unittest.SkipTest('[{}] not available'.format('thread_cpu_clock')) 

        obj = comp.thread_cpu_clock("thread_cpu_clock")
        #obj.push()
        obj.start()
        time.sleep(1)
        obj.stop()
        #obj.pop()

        print("\n[{}]> result: {}".format(self.shortDescription(), obj.get_labeled()))

        # check for near equal
        self.assertAlmostEqual(0.0, obj.get()[0], delta=deltatimer)

# -------------------------------------------------------------------------------------- #
    # test thread cpu utilization
    def test_thread_cpu_util(self):
        """
        thread cpu utilization
        """
        if not check_available(comp.thread_cpu_util):
            raise unittest.SkipTest('[{}] not available'.format('thread_cpu_util')) 

        obj = comp.thread_cpu_util("thread_cpu_util")
        #obj.push()
        obj.start()
        # work for 750ms
        consume(750)
        # sleep for 250ms
        time.sleep(0.25)
        obj.stop()
        #obj.pop()

        print("\n[{}]> result: {}".format(self.shortDescription(), obj.get_labeled()))

        # check for near equal
        self.assertAlmostEqual(75.0, obj.get()[0], delta=deltautil)

# -------------------------------------------------------------------------------------- #
    # test process_cpu_clock
    def test_process_cpu_timer(self):
        """
        process cpu timer 
        """
        if not check_available(comp.process_cpu_clock):
            raise unittest.SkipTest('[{}] not available'.format('process_cpu_clock')) 

        obj = comp.process_cpu_clock("process_cpu_clock")
        #obj.push()
        obj.start()
        time.sleep(1)
        obj.stop()
        #obj.pop()

        print("\n[{}]> result: {}".format(self.shortDescription(), obj.get_labeled()))

        # check for near equal
        self.assertAlmostEqual(0.0, obj.get()[0], delta=deltatimer)

# -------------------------------------------------------------------------------------- #
    # test process cpu utilization
    def test_process_cpu_util(self):
        """
        process cpu utilization
        """
        if not check_available(comp.process_cpu_util):
            raise unittest.SkipTest('[{}] not available'.format('process_cpu_util'))

        obj = comp.process_cpu_util("thread_cpu_util")
        #obj.push()
        obj.start()
        # work for 750ms
        consume(750)
        # sleep for 250ms
        time.sleep(0.25)
        obj.stop()
        #obj.pop()

        print("\n[{}]> result: {}".format(self.shortDescription(), obj.get_labeled()))

        # check for near equal
        self.assertAlmostEqual(75.0, obj.get()[0], delta=deltautil)

# ----------------------------- main test runner ---------------------------------------- #
# test runner
if __name__ == '__main__':
    # run all tests
    unittest.main()
