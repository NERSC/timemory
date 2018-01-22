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

## @file test.py
## A simple test of the timemory module
##

import sys
import os
import timemory as tim
import time
import argparse

default_nfib = 35
tim.enable_signal_detection()

def fibonacci(n):
    if n < 2:
        return n 
    return fibonacci(n-1) + fibonacci(n-2)


def test():
    print ('test: func() {}'.format(tim.FUNC()))
    print ('test: func(2) {}'.format(tim.FUNC(2)))

    
def calcfib(nfib):
    autotimer = tim.auto_timer()
    t = tim.timer("> [pyc] fib({}) ".format(nfib))
    t.start()
    ret = fibonacci(nfib)
    t.stop()
    print ('fibonacci({}) = {}\n'.format(nfib, ret))
    t.report()
    return ret


def main(nfib):
    print ('')
    print ('main: file() {}'.format(tim.FILE()))
    print ('main: line() {}'.format(tim.LINE()))
    print ('main: line() {}'.format(tim.LINE()))
    print ('main: func() {}'.format(tim.FUNC()))
    test()
    print ('')
    tman = tim.timing_manager()
    calcfib(int(nfib))
    tman.report()
    tman.serialize('output.json')
    print ('')
    try:
        tim.util.plot(files=["output.json"], display=False)
    except:
        print ("Error! Unable to plot 'output.json'")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nfib",
                        help="Number of fibonacci calculations",
                        default=35, type=int)
    args = tim.util.add_arguments_and_parse(parser)
    main(args.nfib)
