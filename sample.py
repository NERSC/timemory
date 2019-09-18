#!/usr/bin/env python

import timemory
import time

@timemory.util.auto_timer()
def foo():
    time.sleep(2)
    with timemory.util.auto_timer(""):
        time.sleep(2)

def get_timemory_config():
    return [getattr(timemory.component, x) 
            for x in ["wall_clock", "cpu_clock"]]
            
@timemory.util.auto_tuple(get_timemory_config(), key="")
def bar():
    time.sleep(2)
    with timemory.util.auto_tuple(get_timemory_config(), key=""):
        time.sleep(2)

if __name__ == "__main__":
    foo()
    bar()
