#!@PYTHON_EXECUTABLE@

import os
import time
import json
import unittest
import threading
import inspect
import numpy as np
import timemory as tim
from timemory import components as comp
from timemory.profiler import profile
from timemory.bundle import auto_timer, auto_tuple, marker

# --------------------------- test setup variables ----------------------------------- #


# --------------------------- helper functions ----------------------------------------- #

# check availability of a component
def check_available(component):
    return inspect.isclass(component)

# compute fibonacci
def fibonacci(n):
    return n if n < 2 else (fibonacci(n-1) + fibonacci(n-2))

# sleep for n nanosec
def do_sleep(n):
    time.sleep(n * 1e-9)

# cpu utilization for n nanosec
def consume(n):
    # consume cpu
    now = time.time_ns()
    while(time.time_ns() < (now + n)):
        pass

# get auto_tuple config
def get_config(items=["wall_clock"]):
    return [getattr(tim.component, x) for x in items]

# -------------------------- Thottle Tests set ---------------------------------------- #
# Throttle tests class
class TiMemoryThrottleTests(unittest.TestCase):
    # setup class: timemory settings
    @classmethod
    def setUpClass(self):
        # set up environment variables
        os.environ["TIMEMORY_VERBOSE"] = "1"
        os.environ["TIMEMORY_COLLAPSE_THREADS"] = "OFF"

        tim.settings.parse()
        
        tim.settings.verbose = 1
        tim.settings.debug = False
        tim.settings.json_output = True
        tim.settings.mpi_thread  = False
        tim.settings.file_output = False
        tim.settings.dart_output = True
        tim.settings.dart_count = 1
        tim.settings.banner = False
        
        tim.timemory_trace_init("wall_clock", False, "throttle_tests")

        self.nthreads = 4

    # Tear down class: timemory_finalize
    @classmethod
    def tearDownClass(self):
        # unset environment variables
        del os.environ['TIMEMORY_VERBOSE']
        del os.environ['TIMEMORY_COLLAPSE_THREADS']

        # tim.timemory_trace_finalize()
        # timemory finalize
        #tim.timemory_finalize()
        # tim.dmp.finalize()
        pass

# -------------------------------------------------------------------------------------- #
    # test expect_true
    def test_expect_true(self):
        """
        expect_true
        """
        n = 2 * tim.settings.throttle_count
        tim.timemory_push_trace("true")
        for i in range(n):
            tim.timemory_push_trace(self.shortDescription())
            tim.timemory_pop_trace(self.shortDescription())
        tim.timemory_pop_trace("true")

        self.assertTrue(tim.timemory_is_throttled(self.shortDescription()))

# -------------------------------------------------------------------------------------- #
    # test expect_false
    def test_expect_false(self):
        """
        expect_false
        """
        n = 2 * tim.settings.throttle_count
        v = 2 * tim.settings.throttle_value

        for i in range(n):
            tim.timemory_push_trace(self.shortDescription())
            consume(v)
            tim.timemory_pop_trace(self.shortDescription())

        self.assertFalse(tim.timemory_is_throttled(self.shortDescription()))

# -------------------------------------------------------------------------------------- #
    def test_region_serial(self):
        """
        region_serial
        """
        def _run(name):
            tim.timemory_push_region("rsthread")
            n = 8 * tim.settings.throttle_count
            for i in range(n):
                tim.timemory_push_region(name)
                tim.timemory_pop_region(name)

            tim.timemory_pop_region("rsthread")

        for i in range(self.nthreads):
            _run(self.shortDescription())
        
        #print(tim.timemory_is_throttled(self.shortDescription()))
        #print(tim.timemory_is_throttled("thread"))

# -------------------------------------------------------------------------------------- #
    # test region_multithreaded
    def test_region_multithreaded(self):
        """
        region_multithreaded
        """
        def _run(name):
            tim.timemory_push_region("rthread")
            n = 8 * tim.settings.throttle_count
            for i in range(n):
                tim.timemory_push_region(name)
                tim.timemory_pop_region(name)

            tim.timemory_pop_region("rthread")

        threads = []

        for i in range (self.nthreads):
            thd = threading.Thread(target=_run, args=(self.shortDescription(),))
            thd.start()
            threads.append(thd)

        for itr in threads:
            itr.join()

# -------------------------------------------------------------------------------------- #
    # test multithreaded
    def test_multithreaded(self):
        """
        multithreaded
        """
        # using tuple_t = tim::auto_tuple<tim::component::wall_clock>;

        # np.array of False
        is_throttled = np.full(self.nthreads, False)

        # _run function
        def _run(name, idx):
            tim.timemory_push_trace("mthread")
            n = 2 * tim.settings.throttle_count
            v = 2 * tim.settings.throttle_value
            if (idx % 2 == 1):
                for i in range(n):
                    tim.timemory_push_trace(name)
                    consume(v)
                    tim.timemory_pop_trace(name)
            else:
                for i in range(n):
                    tim.timemory_push_trace(name)
                    tim.timemory_pop_trace(name)

            tim.timemory_pop_trace("mthread")
            is_throttled[idx] = tim.timemory_is_throttled(name)

        # thread handles
        threads = []

        # make new threads
        for i in range (self.nthreads):
            thd = threading.Thread(target=_run, args=(self.shortDescription(), i))
            thd.start()
            threads.append(thd)

        # wait for join
        for itr in threads:
            itr.join()

        # check assertion
        for i in range (self.nthreads):
            _answer = False if (i % 2 == 1) else True
            print("thread " + str(i) + " throttling: " + str(is_throttled[i]))
            self.assertTrue(is_throttled[i] == _answer)

# -------------------------------------------------------------------------------------- #
    # test tuple_serial
    def test_tuple_serial(self):
        """
        tuple_serial
        """
        @marker(components=("wall_clock"), key="thread")
        def _run(name):
            with auto_tuple(get_config(["wall_clock"]), key="auto_tuple_serial"):
                n = 8 * tim.settings.throttle_count
                for i in range(n):
                    with marker(components=("wall_clock"), key=self.shortDescription()):
                        pass
            self.assertFalse(tim.timemory_is_throttled("thread"))
            self.assertFalse(tim.timemory_is_throttled(self.shortDescription()))

        # run with auto tuple (wall_clock)
        for i in range(self.nthreads):
            _run(self.shortDescription())

# -------------------------------------------------------------------------------------- #
    # test tuple_multithreaded
    def test_tuple_multithreaded(self):
        """
        tuple_multithreaded
        """
        @marker(components=("wall_clock"), key="thread")
        def _run(name):
            with auto_tuple(get_config(["wall_clock"]), key="auto_tuple_multithreaded"):
                n = 8 * tim.settings.throttle_count
                for i in range(n):
                    with marker(components=("wall_clock"), key=self.shortDescription()):
                        pass

            self.assertFalse(tim.timemory_is_throttled(self.shortDescription()))
            self.assertFalse(tim.timemory_is_throttled("thread"))

        # thread handles
        threads = []

        # make new threads
        for i in range (self.nthreads):
            thd = threading.Thread(target=_run, args=(self.shortDescription(), ))
            thd.start()
            threads.append(thd)

        # wait for join
        for itr in threads:
            itr.join()

# ----------------------------- main test runner ---------------------------------------- #
# main runner
if __name__ == '__main__':
    # run all tests
    unittest.main()
