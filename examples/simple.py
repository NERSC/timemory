#!/usr/bin/env python

import sys
import os
import timemory as tim
import time

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
    tim.plot.plot(["output.json"], display=False)
    
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(int(sys.argv[1]))
    else:
        main(default_nfib)
