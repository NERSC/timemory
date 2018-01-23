#!/usr/bin/env python

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

## @file options.py
## Options for TiMemory module
##

import sys
import os
from os.path import dirname
from os.path import basename
from os.path import join
import argparse


#------------------------------------------------------------------------------#
def default_max_depth():
    return 65536


#------------------------------------------------------------------------------#
def ensure_directory_exists(file_path):
    # mkdir -p $(basename file_path)
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory) and directory != '':
        os.makedirs(directory)


#------------------------------------------------------------------------------#
#   Default variables
class opts():

    report_file = False
    serial_report = True
    use_timers = True
    max_timer_depth = default_max_depth()
    report_fname = "timing_report.out"
    serial_fname = "timing_report.json"

    def __init__(self):
        pass

    @staticmethod
    def set_report(fname):
        ensure_directory_exists(fname)
        opts.report_fname = fname
        opts.report_file = True


    @staticmethod
    def set_serial(fname):
        ensure_directory_exists(fname)
        opts.serial_fname = fname
        opts.serial_report = True


#------------------------------------------------------------------------------#
def add_arguments(parser, fname=None):
    # Function to add default output arguments
    def get_file_tag(fname):
        _l = basename(fname).split('.')
        _l.pop()
        return ("{}".format('_'.join(_l)))

    def_fname = "timing_report"
    if fname is not None:
        def_fname = '_'.join(["timing_report", get_file_tag(fname)])

    parser.add_argument('--output-dir', required=False,
                        default='./', type=str, help="Output directory")
    parser.add_argument('--filename', required=False,
                        default=def_fname, type=str,
        help="Filename for timing report w/o directory and w/o suffix")
    parser.add_argument('--disable-timers', required=False,
                        action='store_false',
                        dest='use_timers',
                        help="Disable timers for script")
    parser.add_argument('--enable-timers', required=False,
                        action='store_true',
                        dest='use_timers', help="Enable timers for script")
    parser.add_argument('--disable-timer-serialization',
                        required=False, action='store_false',
                        dest='serial_report',
                        help="Disable serialization for timers")
    parser.add_argument('--enable-timer-serialization',
                        required=False, action='store_true',
                        dest='serial_report',
                        help="Enable serialization for timers")
    parser.add_argument('--max-timer-depth',
                        help="Maximum timer depth",
                        type=int,
                        default=65536)

    parser.set_defaults(use_timers=True)
    parser.set_defaults(serial_report=False)


#------------------------------------------------------------------------------#
def parse_args(args):
    # Function to handle the output arguments

    opts.serial_report = args.serial_report
    opts.use_timers = args.use_timers
    opts.max_timer_depth = args.max_timer_depth

    _report_fname = "{}.{}".format(args.filename, "out")
    _serial_fname = "{}.{}".format(args.filename, "json")
    opts.report_fname = join(args.output_dir, _report_fname);
    opts.serial_fname = join(args.output_dir, _serial_fname);

    import timemory as tim
    tim.toggle(opts.use_timers)
    tim.set_max_depth(opts.max_timer_depth)


#------------------------------------------------------------------------------#
def add_arguments_and_parse(parser, fname=None):
    # Combination of timing.add_arguments and timing.parse_args but returns
    add_arguments(parser, fname)
    args = parser.parse_args()
    parse_args(args)
    return args
