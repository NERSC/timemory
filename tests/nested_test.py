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

## @file nested.py
## Auto-timer nesting example
##
## The output of this timing module will be very long. This output demonstrates
## the ability to distiguish the call tree history of the auto-timers from
## a recursive function called in many different contexts but the context
## matching func_1 called from a loop (the output will have lap == 2):
## > [pyc] |_main@'nested.py':86                 :  2.227 wall,  2.100 user +  0.100 system =  2.200 CPU [sec] ( 98.8%) : RSS {tot,self}_{curr,peak} : (121.9|121.9) | ( 69.9| 69.9) [MB]
## > [pyc]   |_func_1@'nested.py':65             :  0.659 wall,  0.600 user +  0.040 system =  0.640 CPU [sec] ( 97.1%) : RSS {tot,self}_{curr,peak} : (115.4|115.4) | ( 63.4| 63.4) [MB] (total # of laps: 2)
## > [pyc]     |_fibonacci@(15)@'nested.py':46   :  0.595 wall,  0.580 user +  0.020 system =  0.600 CPU [sec] (100.8%) : RSS {tot,self}_{curr,peak} : (115.4|115.4) | (  0.9|  0.9) [MB] (total # of laps: 2)
##

import sys
import os
import time
import argparse
import numpy as np
import traceback

import timemory as tim
from timemory import options
from timemory import signals
from timemory import options
from timemory import plotting

tim.enable_signal_detection([signals.sys_signal.Hangup,
                             signals.sys_signal.Interrupt,
                             signals.sys_signal.FPE,
                             signals.sys_signal.Abort ])

array_size = 8000000

#------------------------------------------------------------------------------#
# create an exit action function
def exit_action(errcode):
    tman = tim.timing_manager()
    tim.report(no_min=True)
    fname = 'nested_test_err_{}.out'.format(errcode)
    f = open(fname, 'w')
    f.write('{}\n'.format(tman))
    f.close()


#------------------------------------------------------------------------------#
# set the exit action function
tim.set_exit_action(exit_action)


#------------------------------------------------------------------------------#
# NOTE: Using decorator on a recursive function will produce a very different
# output
def fibonacci(n):
    if n < 2:
        return n 
    autotimer = tim.auto_timer('({})'.format(n), added_args=True)
    return fibonacci(n-1) + fibonacci(n-2)


#------------------------------------------------------------------------------#
@tim.util.auto_timer(add_args=True)
def func_1(n):
    fibonacci(n)


#------------------------------------------------------------------------------#
@tim.util.auto_timer(add_args=True)
def func_2(n):
    func_1(n)
    fibonacci(n)


#------------------------------------------------------------------------------#
@tim.util.auto_timer(add_args=True)
def func_3(n):
    func_1(n)
    func_2(n)


#------------------------------------------------------------------------------#
@tim.util.auto_timer("'AUTO_TIMER_DECORATOR_KEY_TEST':{}".format(tim.LINE()))
def main(nfib):
    for i in range(2):
        func_1(nfib)
    func_2(nfib)
    func_3(nfib)


#------------------------------------------------------------------------------#
if __name__ == "__main__":

    t = tim.timer("Total time")
    t.start()
    print ('')

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nfib",
                        help="Number of fibonacci calculations",
                        default=15, type=int)
    parser.add_argument("-s", "--size",
                        help="Size of array allocations",
                        default=array_size, type=int)
    args = options.add_arguments_and_parse(parser)
    array_size = args.size

    options.set_report("nested_report.out")
    options.set_serial("nested_report.json")

    rss = tim.rss_usage()
    rss.record()

    try:
        main(args.nfib)
        print ('\nTiming manager size: {}\n'.format(tim.size()))
        tman = tim.timing_manager()
        tman -= rss
        tman.report()
        tman.serialize('output.json')
        print ('')
        _data = tim.plotting.read(tman.json())
        _data.title = tim.FILE(noquotes=True)
        _data.filename = options.serial_fname
        plotting.plot(data = [_data], files = ["output.json"], output_dir=options.output_dir)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=5)
        print ('Exception - {}'.format(e))

    t.stop()
    print("\nRSS usage at initialization: {}".format(rss))
    t -= rss
    print ('')
    t.report()
    print ('')
    print("RSS usage at finalization: {}\n".format(tim.rss_usage(record=True)))
