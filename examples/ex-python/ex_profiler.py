#!@PYTHON_EXECUTABLE@

import timemory
from timemory.profiler import profile
from timemory.util import auto_timer
import sys


def fib(n):
    return n if n < 2 else (fib(n-1) + fib(n-2))


@profile(["wall_clock", "peak_rss"])
def run_profile(nfib):
    return fib(nfib)


if __name__ == "__main__":
    nfib = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    ans = run_profile(nfib)
    print("Answer = {}".format(ans))
    timemory.finalize()
