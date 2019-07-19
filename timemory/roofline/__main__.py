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
import traceback

import timemory
import timemory.roofline as _roofline


def plot():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-ai", "--arithmetic-intensity", type=str, help="AI intensity input")
        parser.add_argument("-op", "--operations", type=str, help="Operations input")
        parser.add_argument("-d", "--display", action='store_true', help="Display plot")
        parser.add_argument("-o", "--output-file", type=str, help="Output file",
                            default="roofline")
        parser.add_argument("-D", "--output-dir", type=str, help="Output directory",
                            default=os.getcwd())
        parser.add_argument("--format", type=str, help="Image format", default="png")

        args = parser.parse_args()

        fname = os.path.basename(args.output_file)
        fdir = os.path.realpath(args.output_dir)

        fai = open(args.arithmetic_intensity, 'r')
        fop = open(args.operations, 'r')
        _roofline.plot_roofline(json.load(fai), json.load(fop), args.display,
                                args.output_file, args.format, args.output_dir)

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=5)
        print('Exception - {}'.format(e))
        sys.exit(1)

    print('Done - {}'.format(sys.argv[0]))
    sys.exit(0)


if __name__ == "__main__":
    plot()
