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

## @file nested.py
## Auto-timer nesting example
##

import sys
import os
import timemory as tim
import time
import argparse
import numpy as np

tim.enable_signal_detection()

array_size = 8000000


#------------------------------------------------------------------------------#
def fibonacci(n):
    if n < 2:
        return n 
    autotimer = tim.auto_timer('({})@{}:{}'.format(n, tim.FILE(), tim.LINE()))
    return fibonacci(n-1) + fibonacci(n-2)


#------------------------------------------------------------------------------#
def alloc_memory(n=array_size):
    autotimer = tim.auto_timer('{}:{}'.format(tim.FILE(), tim.LINE()))
    arr = np.ones(shape=[n], dtype=float)
    return arr


#------------------------------------------------------------------------------#
def func_mem(n=array_size):
    autotimer = tim.auto_timer('{}:{}'.format(tim.FILE(), tim.LINE()))
    oparr = alloc_memory(n)


#------------------------------------------------------------------------------#
def func_1(n):
    autotimer = tim.auto_timer('{}:{}'.format(tim.FILE(), tim.LINE()))
    fibonacci(n)
    func_mem(array_size)


#------------------------------------------------------------------------------#
def func_2(n):
    autotimer = tim.auto_timer('{}:{}'.format(tim.FILE(), tim.LINE()))
    func_1(n)
    fibonacci(n)


#------------------------------------------------------------------------------#
def func_3(n):
    autotimer = tim.auto_timer('{}:{}'.format(tim.FILE(), tim.LINE()))
    func_1(n)
    func_2(n)


#------------------------------------------------------------------------------#
def main(nfib):
    autotimer = tim.auto_timer('{}:{}'.format(tim.FILE(), tim.LINE()))
    for i in range(2):
        func_1(nfib)
    func_2(nfib)
    func_3(nfib)


#------------------------------------------------------------------------------#
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nfib",
                        help="Number of fibonacci calculations",
                        default=15, type=int)
    parser.add_argument("-s", "--size",
                        help="Size of array allocations",
                        default=array_size, type=int)
    args = tim.util.add_arguments_and_parse(parser)
    array_size = args.size

    print ('')
    try:
        t = tim.timer("Total time")
        t.start()
        main(args.nfib)
        print ('')
        tman = tim.timing_manager()
        tman.report()
        tman.serialize('output.json')
        print ('')
        tim.util.plot(["output.json"], display=False)
        t.stop()
        print ('')
        t.report()
    except Exception as e:
        print (e.what())
        print ("Error! Unable to plot 'output.json'")

    print ('')
