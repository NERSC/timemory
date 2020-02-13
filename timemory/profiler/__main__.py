#!@PYTHON_EXECUTABLE@
#
# MIT License
#
# Copyright (c) 2018, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S. Dept. of Energy).  All rights reserved.
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

''' @file __main__.py
Command line execution for profiler
'''

import os
import sys
import json
import argparse
import warnings
import traceback
import multiprocessing as mp

import timemory
from .profiler import profile


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help="{} [OPTIONS [OPTIONS...]] -- <OPTIONAL COMMAND TO EXECUTE>".format(sys.argv[0]))
    parser.add_argument("-c", "--components", nargs='+',
                        default=["wall_clock"], help="List of components")
    parser.add_argument("-d", "--output-dir", default=None, type=str,
                        help="Output path of results")
    return parser.parse_known_args()


def run(args, cmd):
    if len(cmd) == 0:
        return

    def get_value(env_var, default_value, dtype, arg=None):
        if arg is not None:
            return dtype(arg)
        else:
            val = os.environ.get(env_var)
            if val is None:
                os.environ[env_var] = "{}".format(default_value)
                return dtype(default_value)
            else:
                return dtype(val)

    output_path = get_value("TIMEMORY_OUTPUT_PATH",
                            "timemory-output", str, args.output_dir)
    timemory.settings.output_path = output_path

    p = profile(args.components)

    progname = cmd[0]
    sys.path.insert(0, os.path.dirname(progname))
    with open(progname, 'rb') as fp:
        code = compile(fp.read(), progname, 'exec')

    import __main__
    dict = __main__.__dict__
    print("code: {} {}".format(type(code).__name__, code))
    globs = {
        '__file__': progname,
        '__name__': '__main__',
        '__package__': None,
        '__cached__': None,
        **dict
    }

    p.runctx(code, globs, None)


def run_profiler():
    args, argv = parse_args()

    print("args: {}".format(args))
    print("argv: {}".format(argv))

    if len(argv) < 2 or (len(argv) > 1 and argv[0] != "--"):
        raise RuntimeError("Specify command to run with -- <command-to-run>")

    try:
        sys.argv = argv[1:]
        run(args, argv[1:])

    except Exception as e:
        import traceback
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=10)
        print('Exception - {}'.format(e))


if __name__ == "__main__":
    run_profiler()
