#!@PYTHON_EXECUTABLE@

import sys
from timemory.util import auto_timer


def fib(n):
    return n if n < 2 else (fib(n-1) + fib(n-2))


@auto_timer("total", mode="full")
def main(nfib):

    ans = fib(nfib)
    with auto_timer("nested"):
        ans += fib(nfib + 1)

    print("Answer = {}".format(ans))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 34)
