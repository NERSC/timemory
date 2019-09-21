#!@PYTHON_EXECUTABLE@

import sys
import timemory
import numpy as np


def fib(n):
    return n if n < 2 else (fib(n-1) + fib(n-2))


if __name__ == "__main__":
    ans = 0
    with timemory.util.auto_timer(key=sys.argv[0].strip("./"), mode="blank"):
        ans += fib(34)
        with timemory.util.auto_timer(key=sys.argv[0].strip("./"), mode="blank"):
            ans += fib(34)
    print("fibonacci(34) = {}".format(fib(34)))
