#!/usr/bin/env python

import timemory
import json
import time

nfib = 33


def fibonacci(n):
    """
    Use this function to get CPU usage
    """
    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)


#
#   CONFIG
#
def get_timemory_config(config=""):
    """
    Provides auto-tuple configurations
    """
    configs = {}
    configs[None] = ["wall_clock", "cpu_clock"]
    items = configs[None] if config not in configs else configs[config]
    #
    # generate the list of enums
    #
    return [getattr(timemory.component, x) for x in items]


#
#   AUTO-TIMER
#
@timemory.util.auto_timer()
def foo():
    """
    Demonstrate decorator and context-manager with auto_timer
    Sleep for 2 seconds then run fibonacci calculation within context-manager
    """
    time.sleep(2)
    with timemory.util.auto_timer(key="[fibonacci]"):
        print("fibonacci({}) = {}".format(nfib, fibonacci(nfib)))


#
#   AUTO-TUPLE
#
@timemory.util.auto_tuple(get_timemory_config(), key="")
def bar():
    """
    Demonstrate decorator and context-manager with auto_tuple
    Run fibonacci calculation and then sleep for 2 seconds within context-manager
    """
    print("fibonacci({}) = {}".format(nfib, fibonacci(nfib)))
    with timemory.util.auto_tuple(get_timemory_config(), key="[sleep]"):
        time.sleep(2)


#
#   EXECUTE
#
if __name__ == "__main__":

    # do some work
    foo()
    bar()

    # get the dictionary
    data = timemory.get()
    # print the dictionary
    print("TIMEMORY_DATA:\n{}".format(json.dumps(data, indent=4, sort_keys=True)))
