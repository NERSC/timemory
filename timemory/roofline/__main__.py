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
Command line execution for roofline plotting library
'''

import os
import sys
import json
import argparse
import warnings
import traceback
import multiprocessing as mp

import timemory
import timemory.roofline as _roofline



def parse_args(add_run_args=False):
    parser = argparse.ArgumentParser(add_help=False, prog='timemory.roofline',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog ='''\
Examples:
  - Perform a Roofline analysis with default parameters
        python -m timemory.roofline -- ./application
  - Perform a Roofline analysis with user-defined dimensions: 1000x800, DPI = 50
        python -m timemory.roofline -P 1000 800 50 -- ./application
                                     ''')

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help="{} [OPTIONS [OPTIONS...]] -- <OPTIONAL COMMAND TO EXECUTE>".format(sys.argv[0]))
    parser.add_argument("-d", "--display",
                        action='store_true', help="Display plot")
    parser.add_argument("-o", "--output-file", type=str, help="Output file",
                        default="roofline")
    parser.add_argument("-D", "--output-dir", type=str, help="Output directory",
                        default=os.getcwd())
    lbandwidth = ["l1", "l2", "l3", "dram"]
    parser.add_argument("-b", "--bandwidth", type=str, help='Bandwidth type - available optionss are: '
                        +', '.join(lbandwidth)+", \"dram\" as default.", choices=lbandwidth, metavar='',
                        dest="bandwidths", default=['dram'])
    parser.add_argument("-tb", "--txn_bandwidth", type=float, metavar='',
                        help="GPU Instruction Roofline transaction bandwidth peak: L1, L2, DRAM. "
                        + "Three arguments are expected. \"437.5 93.6 25.9\" (NVIDIA V100 values)"
                        + " as default.", dest="txns_bandwidths", default=[437.5, 93.6, 25.9], nargs=3)
    parser.add_argument("-iP", "--inst_peak", type=float,
                        help="GPU Instruction peak (per warp) in GIPS (NVIDIA V100 peak set by default)",
                        dest='inst_peak', default=[489.60], nargs=1)
    parser.add_argument("--format", type=str,
                        help="Image format", default="png")
    parser.add_argument("-T", "--title", type=str, dest='title',
                        help="Title for the plot, \"Roofline\" as default.", default="Roofline")
    parser.add_argument("-P", "--plot-dimensions", type=int, metavar='',
                        help="Image dimensions: Width, Height, DPI. Three arguments are expected."
                        + " \"1600 1200 100\" as default.", default=[1600, 1200, 100], nargs=3)
    parser.add_argument("-R", "--rank", type=int,
                        help="MPI Rank", default=None)
    parser.add_argument("-v", "--verbose", type=int,
                        help="Verbosity", default=None)
    parser.add_argument("-e", "--echo-dart", action='store_true',
                        help="Echo image as DartMeasurementFile")
    ltype = ["cpu_roofline", "cpu_roofline_sp",
             "cpu_roofline_dp", "gpu_roofline", "gpu_roofline_hp",
             "gpu_roofline_sp", "gpu_roofline_dp", "gpu_roofline_inst"]
    parser.add_argument("-t", "--rtype", type=str, choices=ltype,
                        help='Roofline type - available optionss are: '+', '.join(ltype) +
                        ", \"cpu_roofline\" as default.", metavar='', default="cpu_roofline")

    if add_run_args:
        parser.add_argument("-p", "--preload", help="Enable preloading libtimemory.so",
                            action='store_true')
        parser.add_argument("-k", "--keep-going", help="Continue despite execution errors",
                            action='store_true')
        parser.add_argument("-r", "--rerun", help="Re-run this mode and not the other", type=str,
                            choices=["ai", "op"], default=None)
        parser.add_argument("-n", "--num-threads", help="Set the number of threads for peak calculation",
                            default=None, type=int)
    else:
        parser.add_argument("-ai", "--arithmetic-intensity",
                            type=str, help="AI intensity input")
        parser.add_argument("-op", "--operations",
                            type=str, help="Operations input")

    return parser.parse_args()


def plot(args):

    try:
        fname = os.path.basename(args.output_file)
        fdir = os.path.realpath(args.output_dir)

        fai = open(args.arithmetic_intensity, 'r')
        fop = open(args.operations, 'r')

        ai_data = json.load(fai)
        op_data = json.load(fop)

        ai_ranks = ai_data["timemory"]["ranks"]
        op_ranks = op_data["timemory"]["ranks"]

        band_labels = [element.upper() for element in args.bandwidths]

        if len(ai_ranks) != len(op_ranks):
            raise RuntimeError("Number of ranks in output files is different: {} vs. {}".format(
                len(ai_ranks), len(op_ranks)))

        if len(op_data) == 1:
            _roofline.plot_roofline(ai_ranks[0], op_ranks[0], band_labels,
                                    args.txns_bandwidths, args.inst_peak, args.rtype,
                                    args.display, fname, args.format, fdir, args.title,
                                    args.plot_dimensions[0], args.plot_dimensions[1],
                                    args.plot_dimensions[2], args.echo_dart)
        else:
            _rank = 0
            for _ai, _op in zip(ai_ranks, op_ranks):
                _fname = "{}_{}".format(fname, _rank)
                _title = "{} (MPI rank: {})".format(args.title, _rank)
                _roofline.plot_roofline(_ai, _op, band_labels,
                                        args.txns_bandwidths, args.inst_peak, args.rtype,
                                        args.display, _fname, args.format, fdir, _title,
                                        args.plot_dimensions[0], args.plot_dimensions[1],
                                        args.plot_dimensions[2], args.echo_dart)
                _rank += 1

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=5)
        print('Exception - {}'.format(e))
        sys.exit(1)

    print('Done - {}'.format(sys.argv[0]))
    sys.exit(0)


def run(args, cmd):
    if len(cmd) == 0:
        return

    def get_environ(env_var, default_value, dtype):
        val = os.environ.get(env_var)
        if val is None:
            os.environ[env_var] = "{}".format(default_value)
            return dtype(default_value)
        else:
            return dtype(val)

    output_path = get_environ("TIMEMORY_OUTPUT_PATH", "timemory-output", str)
    output_prefix = get_environ("TIMEMORY_OUTPUT_PREFIX", "", str)
    os.environ["TIMEMORY_JSON_OUTPUT"] = "ON"

    if args.num_threads is not None:
        os.environ["TIMEMORY_ROOFLINE_NUM_THREADS"] = "{}".format(
            args.num_threads)

    if args.preload:
        # walk back up the tree until we find libtimemory-preload.<EXT>
        preload = os.path.realpath(os.path.dirname(__file__))
        libname = None
        preload_env = None

        import platform
        if platform.system() == "Linux":
            libname = "libtimemory.so"
            preload_env = "LD_PRELOAD"
        elif platform.system() == "Darwin":
            libname = "libtimemory.dylib"
            preload_env = "DYLD_INSERT_LIBRARIES"
            os.environ["DYLD_FORCE_FLAT_NAMESPACE"] = "1"

        # platform support pre-loading
        if libname is not None:
            for i in range(0, 5):
                if os.path.exists(os.path.join(preload, libname)):
                    preload = os.path.join(preload, libname)
                    break
                else:
                    preload = os.path.dirname(preload)

        if preload_env is not None:
            current_preload = os.environ.get(preload_env)
            if current_preload is not None:
                os.environ[preload_env] = "{}:{}".format(
                    current_preload, preload)
            else:
                os.environ[preload_env] = "{}".format(preload)
        elif libname is not None:
            print("Warning! Unable to locate '{}'. Preloading failed".format(libname))

    def handle_error(ret, cmd, keep_going):
        err_msg = "Error executing: '{}'".format(" ".join(cmd))
        if ret != 0 and not keep_going:
            raise RuntimeError(err_msg)
        elif ret != 0 and keep_going:
            barrier = "="*80
            err_msg = "\n\n" + barrier + "\n\n    ERROR: " + \
                err_msg + "\n\n" + barrier + "\n\n"
            sys.stderr.write(err_msg)
            sys.stderr.flush()

    import subprocess as sp

    if args.rerun is None or args.rerun == "ai":
        os.environ["TIMEMORY_ROOFLINE_MODE"] = "ai"
        os.environ["TIMEMORY_ROOFLINE_MODE_CPU"] = "ai"
        os.environ["TIMEMORY_ROOFLINE_MODE_GPU"] = "ai"
        p = sp.Popen(cmd)
        ret = p.wait()
        handle_error(ret, cmd, args.keep_going)

    if args.rerun is None or args.rerun == "op":
        os.environ["TIMEMORY_ROOFLINE_MODE"] = "op"
        os.environ["TIMEMORY_ROOFLINE_MODE_CPU"] = "op"
        os.environ["TIMEMORY_ROOFLINE_MODE_GPU"] = "op"
        p = sp.Popen(cmd)
        ret = p.wait()
        handle_error(ret, cmd, args.keep_going)

    _rank = "" if args.rank is None else "_{}".format(args.rank)
    if "gpu_roofline" in args.rtype:
        args.arithmetic_intensity = os.path.join(
            output_path, "{}{}_activity.json".format(output_prefix, args.rtype))
        args.operations = os.path.join(
            output_path, "{}{}_counters.json".format(output_prefix, args.rtype))
    else:
        args.arithmetic_intensity = os.path.join(
            output_path, "{}{}_ai.json".format(output_prefix, args.rtype))
        args.operations = os.path.join(
            output_path, "{}{}_op.json".format(output_prefix, args.rtype))


def try_plot():
    try:
        # look for "--" and interpret anything after that
        # to be a command to execute
        _argv = []
        _cmd = []

        _argsets = [_argv, _cmd]
        _i = 0
        _separator = '--'

        for _arg in sys.argv[1:]:
            if _arg == _separator:
                _i += 1
                if _i >= len(_argsets):
                    sys.exit(
                        "ERROR: Too many \"{}\" separators provided "
                        "(expected at most {}).".format(_separator,
                                                        len(_argsets) - 1))
            else:
                _argsets[_i].append(_arg)

        sys.argv[1:] = _argv

        args = parse_args(len(_cmd) != 0)
        run(args, _cmd)
        if args.verbose is not None:
            _roofline.VERBOSE = args.verbose
        plot(args)

    except Exception as e:
        msg = "\nCommand line argument error:\n\t{}\n".format(e)
        warnings.warn(msg)


if __name__ == "__main__":
    try_plot()
