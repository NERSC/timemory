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

## @file simple_test.py
## A simple test of the timemory module
##

import sys
import os
import time
import argparse
import traceback

import timemory
from timemory import options
from timemory import plotting


# ---------------------------------------------------------------------------- #
@timemory.util.auto_timer(add_args=True)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)


# ---------------------------------------------------------------------------- #
class Fibonacci(object):

    def __init__(self, n):
        self.n = n

    @timemory.util.auto_timer()
    def calculate(self):
        t = timemory.timer("> [pyc] fib({}) ".format(self.n))
        t.start()
        ret = fibonacci(self.n)
        t.stop()
        print ('fibonacci({}) = {}\n'.format(self.n, ret))
        t.report()
        return ret


# ---------------------------------------------------------------------------- #
def test():
    print ('test: func() {}'.format(timemory.FUNC()))
    print ('test: func(2) {}'.format(timemory.FUNC(2)))


# ---------------------------------------------------------------------------- #
def main(nfib):
    timemory.set_max_depth(5)
    print ('')
    print ('main: file() {}'.format(timemory.FILE()))
    print ('main: line() {}'.format(timemory.LINE()))
    print ('main: line() {}'.format(timemory.LINE()))
    print ('main: func() {}'.format(timemory.FUNC()))
    test()
    print ('')
    tman = timemory.manager()
    fib = Fibonacci(int(nfib))
    ret = fib.calculate()

    timemory.report()
    tman.report(ign_cutoff=True)
    _jsonf = os.path.join(options.output_dir, 'simple_output.json')
    tman.serialize(_jsonf)
    print ('')
    plotting.plot(files=[_jsonf], display=False, output_dir=options.output_dir)

    
# ---------------------------------------------------------------------------- #
def run_test():

    default_nfib = 29
    timemory.enable_signal_detection()

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nfib",
                        help="Number of fibonacci calculations",
                        default=default_nfib, type=int)
    args = options.add_arguments_and_parse(parser)
    timemory.options.output_dir = "test_output"

    main(args.nfib)

    timemory.disable_signal_detection()
    print('"{}" testing finished'.format(__file__))


# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    try:
        run_test()

        if options.ctest_notes:
            manager = timemory.manager()
            f = manager.write_ctest_notes(directory="test_output/simple_test")
            print('"{}" wrote CTest notes file : {}'.format(__file__, f))

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=5)
        print ('Exception - {}'.format(e))
        raise
