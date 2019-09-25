#!@PYTHON_EXECUTABLE@

import sys
import timemory
from timemory.util import auto_timer


def fib(n):
    return n if n < 2 else (fib(n-1) + fib(n-2))


if __name__ == "__main__":

    nfib = int(sys.argv[1]) if len(sys.argv) > 1 else 34

    @auto_timer("%s_%i" % (sys.argv[0], nfib+1), mode="blank")
    def run_fib(n):
        return fib(n)

    ans = 0
    with auto_timer("%s_%i" % (sys.argv[0], nfib)):
        ans += fib(nfib)
        ans += run_fib(nfib + 1)

    print("Answer = {}".format(ans))
