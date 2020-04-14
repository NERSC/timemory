#!@PYTHON_EXECUTABLE@

import sys
import json
import timemory
from timemory.profiler import profile
from timemory.bundle import auto_timer, auto_tuple, marker

def get_config(items=["wall_clock", "cpu_clock"]):
    return [getattr(timemory.component, x) for x in items]

def fib(n):
    return n if n < 2 else (fib(n-1) + fib(n-2))

@profile(["wall_clock", "peak_rss"])
def run_profile(n):
    ''' Run full profiler '''
    return fib(int(n))

@auto_timer()
def run_auto_timer(n):
    ''' Decorator and context manager for high-level pre-defined collection '''
    fib(n)
    with auto_timer(key="auto_timer_ctx_manager"):
        fib(n)

@marker(["wall_clock", "peak_rss"])
def run_marker(n):
    ''' Decorator and context manager for high-level custom collection '''
    fib(n)
    with auto_tuple(get_config(), key="auto_tuple_ctx_manager"):
        fib(n)

if __name__ == "__main__":
    n = int(sys.argv[1] if len(sys.argv) > 1 else 10)

    run_profile(n)
    run_auto_timer(n)
    run_marker(n)

    # Get the results as dictionary
    data = timemory.get()

    print("\n{}".format(json.dumps(data, indent=4, sort_keys=True)))
    # Generate output
    timemory.finalize()
