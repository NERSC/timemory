#!@PYTHON_EXECUTABLE@

import os
import time
import mmap
import numpy as np
import unittest
import threading
import inspect
import random
import timemory as tim
from timemory import components as comp

# --------------------------- test setup variables ----------------------------------- #
niter       = 20
nelements   = 0.95 * (mmap.PAGESIZE * 500)
memory_unit = (tim.units.kilobyte, "KiB")

peak = comp.peak_rss("peak_rss")
curr = comp.page_rss("curr_rss")
current_peak = comp.current_peak_rss("curr_peak_rss")
rb = comp.read_bytes("read_bytes")
wb = comp.written_bytes("written_bytes")

sizeof_int64_t = np.ones(1, np.int64).itemsize
sizeof_double = np.ones(1, np.float64).itemsize

tot_size = nelements * sizeof_int64_t/ memory_unit[0]
tot_rw   = nelements * sizeof_double / memory_unit[0]

peak_tolerance = 5 * tim.units.megabyte
curr_tolerance = 5 * tim.units.megabyte
byte_tolerance = tot_rw

# --------------------------- helper functions ----------------------------------------- #

# compute fibonacci
def fibonacci(n):
    return n if n < 2 else (fibonacci(n-1) + fibonacci(n-2))

# check availability of a component
def check_available(component):
    return inspect.isclass(component)

# get a random entry
def random_entry(vect):
    random.seed(time.time_ns())
    num = int(random.uniform(0, len(vect) - 1))
    return vect[num]

# allocate memory
def allocate():
    peak.reset()
    curr.reset()
    current_peak.reset()

    curr.start()
    peak.start()
    current_peak.start()

    v = np.full(int(nelements), 15, dtype=np.int64)
    ret  = fibonacci(0)
    nfib = random_entry(v)

    for i in range(niter):
        nfib = random_entry(v)
        ret += fibonacci(nfib)
    

    if(ret < 0):
        print("fibonacci({}) * {} = {}\n".format(nfib, niter, ret))

    current_peak.stop()
    curr.stop()
    peak.stop()

# read and write to file
def read_write():

    rb.start()
    wb.start()

    ofs = open("/tmp/file.dat", "wb")

    writeArray = np.array(range(int(nelements)))
    writeArray = writeArray.astype(np.float64)
    
    # make a new bytes array
    writeByteArray = bytearray(writeArray)
    
    # write the data to file
    ofs.write(writeByteArray)

    # close the file
    ofs.close()
    
    # read the data from file
    data = np.fromfile("/tmp/file.dat", dtype=np.float64)

    rb.stop()
    wb.stop()

# get_info for wb and rb
def get_info_rb_wb(obj):
    _unit = memory_unit[0]
    ss = "value = " 
    ss += str(obj.get()[0][0]) + " "
    ss += memory_unit[1] + ", accum = "
    ss += str(obj.get()[0][0]) + " "
    ss += memory_unit[1] + "\n"
    return ss

# get_info for current peak
def get_info_curr_peak(obj):
    _unit = memory_unit[0]
    ss = "value = " 
    ss += str(obj.get()[0][1] - obj.get()[0][0]) + " "
    ss += memory_unit[1] + ", accum = "
    ss += str(obj.get()[0][1] - obj.get()[0][0]) + " "
    ss += memory_unit[1] + "\n"
    return ss

# get_info for peak, page rss
def get_info(obj):
    _unit = memory_unit[0]
    ss = "value = "
    ss += str(obj.get()[0]) + " " 
    ss += str(memory_unit[1]) + ", accum = "
    ss += str(obj.get()[0]) + " " 
    ss += str(memory_unit[1] + "\n")
    return ss

# print_info for peak, page rss
def print_info(name, obj, expected):
    print("")
    print("[" + name + "]>  measured :", obj.get()[0])
    print("[" + name + "]>  expected :", expected, memory_unit[1])
    print("[" + name + "]> data info :", get_info(obj))

# print_info for wb and rb
def print_info_wb_rb(name, obj, expected):
    print("")
    print("[" + name + "]>  measured :", obj.get()[0][0])
    print("[" + name + "]>  expected :", expected, memory_unit[1])
    print("[" + name + "]> data info :", get_info_rb_wb(obj))

# print_info for current peak
def print_info_curr_peak(name, obj, expected):
    print("")
    print("[" + name + "]>  measured :", obj.get()[0][1]-obj.get()[0][0])
    print("[" + name + "]>  expected :", expected, memory_unit[1])
    print("[" + name + "]> data info :", get_info_curr_peak(obj))

# --------------------------- RUsage Tests set ---------------------------------------- #
# RSS usage tests class
class TiMemoryRUsageTests(unittest.TestCase):
    # setup class: timemory settings
    @classmethod
    def setUpClass(self):
        tim.settings.verbose = 0
        tim.settings.debug = False
        tim.settings.json_output = True
        tim.settings.precision = 9
        tim.settings.memory_units = memory_unit[1]
        tim.settings.mpi_thread  = False
        tim.settings.file_output = False
        
        # perform allocation once here
        allocate()

        # perform read write once here
        read_write()

        tim.settings.dart_output = True
        tim.settings.dart_count = 1
        tim.settings.banner = False
        tim.settings.dart_type = "peak_rss"

    # tear down class: timemory_finalize
    @classmethod
    def tearDownClass(self):
        # timemory finalize
        #tim.timemory_finalize()
        # tim.dmp.finalize()
        pass

# -------------------------------------------------------------------------------------- #
    # test peak rss
    def test_peak_rss(self):
        """
        peak rss
        """
        if not check_available(comp.peak_rss):
            raise unittest.SkipTest('[{}] not available'.format('peak_rss')) 

        print_info(self.shortDescription(), peak, tot_size)
        self.assertAlmostEqual(tot_size, peak.get()[0], delta=peak_tolerance)

# -------------------------------------------------------------------------------------- #
    # test page rss
    def test_page_rss(self):
        """
        page rss
        """
        if not check_available(comp.page_rss):
            raise unittest.SkipTest('[{}] not available'.format('page_rss')) 

        print_info(self.shortDescription(), curr, tot_size)
        self.assertAlmostEqual(tot_size, curr.get()[0], delta=curr_tolerance)

# -------------------------------------------------------------------------------------- #
# test read_bytes
    def test_read_bytes(self):
        """
        read bytes
        """
        if not check_available(comp.read_bytes):
            raise unittest.SkipTest('[{}] not available'.format('read_bytes')) 

        print_info_wb_rb(self.shortDescription(), rb, tot_rw)
        self.assertAlmostEqual(tot_rw, (rb.get()[0])[0], delta=byte_tolerance)

# -------------------------------------------------------------------------------------- #
# test write_bytes
    def test_written_bytes(self):
        """
        written bytes
        """
        if not check_available(comp.written_bytes):
            raise unittest.SkipTest('[{}] not available'.format('written_bytes')) 

        print_info_wb_rb(self.shortDescription(), wb, tot_rw)
        self.assertAlmostEqual(tot_rw, wb.get()[0][0], delta=byte_tolerance)

# -------------------------------------------------------------------------------------- #
# test write_bytes
    def test_current_peak_rss(self):
        """
        current peak rss
        """
        if not check_available(comp.current_peak_rss):
            raise unittest.SkipTest('[{}] not available'.format('current_peak_rss')) 

        print_info_curr_peak(self.shortDescription(), current_peak, tot_size)
        self.assertAlmostEqual(tot_size, current_peak.get()[0][1] - current_peak.get()[0][0],
                delta= peak_tolerance)

# ----------------------------- main test runner ---------------------------------------- #
# test runner
if __name__ == '__main__':
    # run all tests
    unittest.main()
