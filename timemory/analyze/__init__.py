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

from __future__ import absolute_import
from __future__ import division

import os
import sys
import json
import argparse
import traceback

__author__ = "Jonathan Madsen"
__copyright__ = "Copyright 2020, The Regents of the University of California"
__credits__ = ["Jonathan Madsen"]
__license__ = "MIT"
__version__ = "@PROJECT_VERSION@"
__maintainer__ = "Jonathan Madsen"
__email__ = "jrmadsen@lbl.gov"
__status__ = "Development"

from .analyze import (
    load,
    match,
    search,
    expression,
    sort,
    group,
    add,
    subtract,
    unify,
    dump,
)


def embedded_analyze(
    _argv, _call_exit=False, _verbose=os.environ.get("TIMEMORY_VERBOSE", 0)
):
    """This is intended to be called from the embedded python interpreter"""
    _call_exit = False
    if _argv is None:
        _argv = sys.argv
        _call_exit = True

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "files",
            metavar="file",
            type=str,
            nargs="+",
            help="Files to analyze",
        )
        parser.add_argument(
            "-f",
            "--format",
            type=str,
            help="Data output format.",
            choices=(
                "dot",
                "flamegraph",
                "tree",
                "table",
                "markdown",
                "html",
                "markdown_grid",
            ),
            default=None,
        )
        parser.add_argument(
            "-o",
            "--output",
            type=str,
            help="Output to file(s)",
            nargs="*",
            default=None,
        )
        parser.add_argument(
            "-M",
            "--mode",
            help="Analysis mode",
            nargs="*",
            choices=("add", "subtract", "unify", "group"),
            default=[],
        )
        parser.add_argument(
            "-m",
            "--metric",
            "--column",
            type=str,
            help="Metric(s) to extract",
            default="sum.inc",
        )
        parser.add_argument(
            "-s",
            "--sort",
            type=str,
            help="Sort the metric",
            choices=("ascending", "descending"),
            default=None,
        )
        parser.add_argument(
            "-g",
            "--group",
            type=str,
            help="Group by a column name in the dataframe",
            default=None,
        )
        parser.add_argument(
            "--field",
            type=str,
            help="Dataframe column to search/match against",
            default="name",
        )
        parser.add_argument(
            "--search",
            type=str,
            default=None,
            help="Regular expression for re.search(...), i.e. a substring match",
        )
        parser.add_argument(
            "--match",
            type=str,
            default=None,
            help="Regular expression for re.match(...), i.e. a full string match",
        )
        parser.add_argument(
            "--expression",
            type=str,
            default=None,
            help=(
                "A space-delimited comparison operation expression using 'x' "
                + "for the variable, numerical values, and: < <= > >= && ||. "
                + "E.g. 'x > 1.0e3 && x < 100000'. 'x' will be -m/--metric"
            ),
        )
        parser.add_argument(
            "-e",
            "--echo-dart",
            help="echo Dart measurement for CDash",
            action="store_true",
        )
        parser.add_argument(
            "--per-thread",
            help="Encode the thread ID in node hash to ensure squashing doesn't combine thread-data",
            action="store_true",
        )
        parser.add_argument(
            "--per-rank",
            help="Encode the rank ID in node hash to ensure squashing doesn't combine rank-data",
            action="store_true",
        )
        parser.add_argument(
            "--select",
            help="Select the component type if the JSON input contains output from multiple components",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--exit-on-failure",
            help="Call exit if failed",
            action="store_true",
        )

        args = parser.parse_args(_argv)

        if args.group is None and args.format is None:
            args.format = "tree"
        if args.group is not None and args.format not in ("table", None):
            raise RuntimeError("Invalid data format for group")

        gfs = []
        for f in args.files:
            gfs.append(
                load(
                    f,
                    select=args.select,
                    per_thread=args.per_thread,
                    per_rank=args.per_rank,
                )
            )

        def apply(_input, _mode, *_args, **_kwargs):
            if _mode == "search":
                return search(_input, *_args, **_kwargs)
            elif _mode == "match":
                return match(_input, *_args, **_kwargs)
            elif _mode == "expression":
                return expression(_input, *_args, **_kwargs)
            elif _mode == "sort":
                return sort(_input, *_args, **_kwargs)
            elif _mode == "add":
                return add(_input)
            elif _mode == "subtract":
                return subtract(_input)
            elif _mode == "unify":
                return unify(_input)
            elif _mode == "group":
                return group(_input, *_args, **_kwargs)

        # apply search before match since this is less restrictive
        if args.search is not None:
            gfs = apply(gfs, "search", pattern=args.search, field=args.field)

        # apply match after search since this is more restrictive
        if args.match is not None:
            gfs = apply(gfs, "match", pattern=args.match, field=args.field)

        # apply numerical expression last
        if args.expression is not None:
            gfs = apply(
                gfs, "expression", math_expr=args.expression, metric=args.metric
            )

        # apply the mutating operations
        if "add" in args.mode:
            gfs = apply(gfs, "add")
        elif "subtract" in args.mode:
            gfs = apply(gfs, "subtract")
        elif "unify" in args.mode:
            gfs = apply(gfs, "unify")

        if args.sort is not None:
            gfs = apply(
                gfs,
                "sort",
                metric=args.metric,
                ascending=(args.sort == "ascending"),
            )

        files = args.output
        if files is not None and len(files) == 1:
            files = files[0]

        if args.group is not None:
            gfs = apply(
                gfs,
                "group",
                metric=args.metric,
                field=args.group,
                ascending=(args.sort == "ascending"),
            )
            args.format = None

        dump(gfs, args.metric, args.format, files, args.echo_dart)

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=10)
        print("Exception - {}".format(e))
        if _call_exit or args.exit_on_failure:
            sys.exit(1)
