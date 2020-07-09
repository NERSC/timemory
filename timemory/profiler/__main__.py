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

PY3 = sys.version_info[0] == 3


# ---------------------------------------------------------------------------- #
# Python 3.x compatibility utils: execfile
try:
    execfile
except NameError:
    # Python 3.x doesn't have 'execfile' builtin
    import builtins
    exec_ = getattr(builtins, "exec")

    def execfile(filename, globals=None, locals=None):
        with open(filename, 'rb') as f:
            exec_(compile(f.read(), filename, 'exec'), globals, locals)


# ---------------------------------------------------------------------------- #
def find_script(script_name):
    """ Find the script.

    If the input is not a file, then $PATH will be searched.
    """
    if os.path.isfile(script_name):
        return script_name
    path = os.getenv('PATH', os.defpath).split(os.pathsep)
    for dir in path:
        if dir == '':
            continue
        fn = os.path.join(dir, script_name)
        if os.path.isfile(fn):
            return fn

    sys.stderr.write('Could not find script %s\n' % script_name)
    raise SystemExit(1)


# ---------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help="{} [OPTIONS [OPTIONS...]] -- <OPTIONAL COMMAND TO EXECUTE>".format(sys.argv[0]))
    parser.add_argument("-c", "--components", nargs='+',
                        default=["wall_clock"], help="List of components")
    parser.add_argument("-d", "--output-dir", default=None, type=str,
                        help="Output path of results")
    parser.add_argument('-b', '--builtin', action='store_true',
                        help="Put 'profile' in the builtins. Use 'profile.enable()' and "
                        "'profile.disable()' in your code to turn it on and off, or "
                        "'@profile' to decorate a single function, or 'with profile:' "
                        "to profile a single section of code.")
    parser.add_argument('-s', '--setup', default=None,
                        help="Code to execute before the code to profile")
    return parser.parse_known_args()


# ---------------------------------------------------------------------------- #
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

# ---------------------------------------------------------------------------- #


def run(prof, cmd):
    if len(cmd) == 0:
        return

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

    prof.runctx(code, globs, None)


# ---------------------------------------------------------------------------- #
def main():
    opts, argv = parse_args()

    # print("opts: {}".format(opts))
    print("argv: {}".format(argv))

    sys.argv[:] = argv
    if opts.setup is not None:
        # Run some setup code outside of the profiler. This is good for large
        # imports.
        setup_file = find_script(opts.setup)
        __file__ = setup_file
        __name__ = '__main__'
        # Make sure the script's directory is on sys.path instead of just
        # kernprof.py's.
        sys.path.insert(0, os.path.dirname(setup_file))
        ns = locals()
        execfile(setup_file, ns, ns)

    from ..libpytimemory import settings, finalize
    from . import Profiler, FakeProfiler

    output_path = get_value("TIMEMORY_OUTPUT_PATH",
                            "timemory-output", str, opts.output_dir)
    settings.output_path = output_path

    # if len(argv) > 1 and argv[0] != "--"):
    #    raise RuntimeError("Specify command to run with -- <command-to-run>")
    script_file = find_script(sys.argv[0])
    __file__ = script_file
    __name__ = '__main__'
    # Make sure the script's directory is on sys.path instead of just
    # kernprof.py's.
    sys.path.insert(0, os.path.dirname(script_file))

    prof = Profiler(opts.components)
    fake = FakeProfiler()

    if PY3:
        import builtins
    else:
        import __builtin__ as builtins

    builtins.__dict__['profile'] = prof
    builtins.__dict__['noprofile'] = fake

    try:
        try:
            if not opts.builtin:
                prof.start()
            execfile_ = execfile
            ns = locals()
            if opts.builtin:
                execfile(script_file, ns, ns)
            else:
                prof.runctx('execfile_(%r, globals())' %
                            (script_file,), ns, ns)
            if not opts.builtin:
                prof.stop()
            finalize()
        except (KeyboardInterrupt, SystemExit):
            pass
    except Exception as e:
        import traceback
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=10)
        print('Exception - {}'.format(e))


# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
