# MIT License
#
# Copyright (c) 2020, The Regents of the University of California,
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

__author__ = "Jonathan Madsen"
__copyright__ = "Copyright 2020, The Regents of the University of California"
__credits__ = ["Jonathan Madsen"]
__license__ = "MIT"
__version__ = "@PROJECT_VERSION@"
__maintainer__ = "Jonathan Madsen"
__email__ = "jrmadsen@lbl.gov"
__status__ = "Development"
__all__ = [
    "load",
    "match",
    "search",
    "expression",
    "sort",
    "group",
    "add",
    "subtract",
    "unify",
    "dump_entity",
    "dump_tree",
    "dump_dot",
    "dump_flamegraph",
    "dump_tabulate",
    "dump_unknown",
    "dump",
]

import os
import sys


def _get_label(_fname):
    """Generates a label for log messages"""
    _lbl = os.path.basename(_fname)
    if "." in _lbl:
        return _lbl[0 : (_lbl.find("."))]  # noqa: E203
    return _lbl


def _get_filename(_file, _fext):
    """Ensures an output file has a certain extension"""
    if _file is None or _fext is None:
        return _file
    else:
        if _fext[0] != ".":
            _fext = f".{_fext}"
        if not _file.endswith(_fext):
            return os.path.splitext(_file)[0] + _fext
    return _file


def _svg_to_png(out, svg_code=None, svg_file=None):
    """Writes an SVG to a PNG"""
    try:
        from cairosvg import svg2png

        lbl = _get_label(out)
        if svg_code is not None:
            print(f"[{lbl}]|0> Outputting '{out}'...")
            svg2png(svg_code, write_to=out)
            return out
        elif svg_file is not None:
            print(f"[{lbl}]|0> Outputting '{out}'...")
            with open(svg_file, "r") as fin:
                svg2png(fin, write_to=out)
        else:
            raise RuntimeError("Error! No svg_code or svg_file")

    except ImportError as e:
        print(e)
        print("Install 'cairosvg' for conversion from SVG to PNG")

    return None


def load(data, *_args, **_kwargs):
    """Loads a graphframe"""
    from timemory import hatchet
    import hatchet as ht

    # print("data: {}".format(data))
    # print("data_type: {}".format(type(data).__name__))
    return ht.GraphFrame.from_timemory(data, *_args, **_kwargs)


def match(data, pattern, field="name"):
    """Find the graphframes which fully match a regular expression"""
    import re

    prog = re.compile(pattern)

    if not isinstance(data, list):
        return data.filter(lambda x: (prog.fullmatch(x[field]) is not None))

    ret = []
    for itr in data:
        ret.append(itr.filter(lambda x: (prog.fullmatch(x[field]) is not None)))
    return ret


def search(data, pattern, field="name"):
    """Find the graphframes with substring that matches regular expression"""
    import re

    prog = re.compile(pattern)

    if not isinstance(data, list):
        return data.filter(lambda x: (prog.search(x[field]) is not None))

    ret = []
    for itr in data:
        ret.append(itr.filter(lambda x: (prog.search(x[field]) is not None)))
    return ret


def expression(data, math_expr, metric="sum.inc"):
    """Find the graphframes with whose value satisfy the given expression

    Arguments:
        data (graphframe or list of graphframes): graphframes to be filtered

        math_expr (str): A space-delimited comparison operation expression using
            'x' for the variable, numerical values, and: < <= > >= && ||

        metric (str): the column in the dataframe to apply the expression to

    Return Value:
        graphframe or list of graphframe: data type will be the same as the input data

    Example:
        The following will filter out any rows whose sum.inc value is less than
        1,000 or greater than 100,000:

            filter(data, 'x > 1.0e3 && x < 1e6', "sum.inc")
    """

    def eval_math_expr(x):
        _expr = math_expr.replace("x", "{}".format(x[metric]))
        # print(f"Executing expression: '{_expr}'")
        _and = _expr.split("&&")
        for itr in _and:
            if "||" in itr:
                _or = itr.split("||")
                _n_or = 0
                for sitr in _or:
                    _n_or += 1 if eval(sitr) else 0
                if _n_or == 0:
                    # print(f'    Failure! {sitr}')
                    return False
            else:
                if not eval(itr):
                    # print(f'    Failure! {itr}')
                    return False
        # print(f'    Success: {_expr}')
        return True

    if math_expr is not None:
        # strictly validate that there is nothing in the expression that
        # is unexpected since `eval(...)` is dangerous to invoke trivially.
        math_split = math_expr.split()
        valid_filter = ("x", "<", "<=", ">", ">=", "&&", "||")
        for token in math_split:
            if token in valid_filter:
                continue
            else:
                try:
                    float(token)
                except ValueError:
                    raise ValueError(
                        f"Error: Token '{token}' is not valid. "
                        + "Must be numeric or one of: {}".format(
                            " ".join(valid_filter)
                        )
                    )
    else:
        raise ValueError("Error: Invalid math expression!")

    if not isinstance(data, list):
        return data.filter(eval_math_expr, num_procs=1)

    ret = []
    for itr in data:
        ret.append(itr.filter(eval_math_expr, num_procs=1))
    return ret


def sort(data, metric="sum", ascending=False):
    """Sort one or more graphframes by a metric"""

    def _generate(itr, _metric, _ascending):
        # try importing from real hatchet
        try:
            from hatchet.graph import Graph
            from hatchet.graphframe import GraphFrame
        except ImportError:
            from timemory.hatchet.graph import Graph  # noqa: F401
            from timemory.hatchet.graphframe import GraphFrame  # noqa: F401

        # generate a new graph from the sorted graphframe
        _data_frame = itr.dataframe.sort_values(
            by=[_metric], ascending=_ascending
        )

        return _data_frame

        # _list_roots = []
        # for node in itr.graph.traverse():
        #     _new = node.copy()
        #     _new.parents.clear()
        #     _new.children.clear()
        #     _list_roots.append(_new)
        # _graph = Graph(_list_roots)
        # _graph.enumerate_traverse()
        # return GraphFrame(_graph, _data_frame, itr.exc_metrics, itr.inc_metrics)

    if not isinstance(data, list):
        return _generate(data, metric, ascending)

    ret = []
    for itr in data:
        ret.append(_generate(itr, metric, ascending))
    return ret


def group(data, metric, field="name", ascending=False):
    """Generate a flat profile"""

    def _generate(itr, _metric, _field, _ascending):
        # Drop all index levels in the DataFrame except ``node``.
        # itr.drop_index_levels()
        return (
            itr.dataframe.groupby(_field)
            .sum()
            .sort_values(by=[_metric], ascending=_ascending)
        )

    if not isinstance(data, list):
        return _generate(data, metric, field, ascending)

    ret = []
    for itr in data:
        ret.append(_generate(itr, metric, field, ascending))
    return ret


def add(data):
    """Adds two or more graphframes"""
    obj = None
    for itr in data:
        if obj is None:
            obj = itr.deepcopy()
        else:
            obj += itr
    return obj


def subtract(data):
    """Subtracts two or more graphframes"""
    obj = None
    for itr in data:
        if obj is None:
            obj = itr.deepcopy()
        else:
            obj -= itr
    return obj


def unify(data):
    """Finds unity between two or more graphframes"""
    obj = None
    for itr in data:
        if obj is None:
            obj = itr.deepcopy()
        else:
            obj.unify(itr.copy())
    return obj


def dump_entity(data, functor, file=None, fext=None):
    """Dumps data to stdout or file.
    file can be file-like or filename.
    """

    def _dump_entity(_data, _file=None):
        if _file is None:
            print(f"{_data}")
            return None
        elif hasattr(_file, "write"):
            _file.write(f"{_data}\n")
            return None
        else:
            # assume filename
            if not os.path.isabs(_file):
                try:
                    from timemory import settings

                    _file = os.path.join(settings.output_prefix, _file)
                except ImportError:
                    pass
            _lbl = _get_label(_file)
            print(f"[{_lbl}]|0> Outputting '{_file}'...")
            with open(_file, "w") as ofs:
                ofs.write(f"{_data}\n")
            return _file

    def _get_entity(_data, _functor):
        return _functor(_data)

    files = []
    if type(data) == type(file):
        for ditr, fitr in zip(data, file):
            _ret = _dump_entity(functor(ditr), _get_filename(fitr, fext))
            if _ret is not None:
                files.append(_ret)
    elif not isinstance(data, list):
        _ret = _dump_entity(functor(data), _get_filename(file, fext))
        if _ret is not None:
            files.append(_ret)
    else:
        _file = _get_filename(file, fext)
        ofs = open(_file, "w") if _file is not None else None
        if ofs is not None:
            lbl = _get_label(_file)
            print(f"[{lbl}]|0> Outputting '{_file}'...")
        for itr in data:
            _dump_entity(functor(itr), ofs)
        if ofs is not None:
            ofs.close()
        if _file is not None:
            files.append(_file)
    return files


def dump_tree(data, metric, file=None, echo_dart=False):
    """Dumps data as a tree to stdout or file"""
    _files = dump_entity(data, lambda x: x.tree(metric), file, ".txt")
    for itr in _files:
        if itr is not None and echo_dart is True:
            from timemory.common import dart_measurement_file

            dart_measurement_file(
                os.path.basename(itr), itr, format="string", type="text"
            )
            # write_ctest_notes(itr)


def dump_dot(data, metric, file=None, echo_dart=False):
    """Dumps data as a dot to stdout or file"""
    from timemory.common import popen, dart_measurement_file, which

    _files = dump_entity(data, lambda x: x.to_dot(metric), file, ".dot")
    for itr in _files:
        if itr is not None:
            lbl = _get_label(itr)
            oitr = _get_filename(itr, ".dot")
            pitr = _get_filename(itr, ".dot.png")
            print(f"[{lbl}]|0> Outputting '{pitr}'...")
            try:
                dot_exe = which("dot")
                if dot_exe is not None:
                    popen(
                        [dot_exe, "-Tpng", f"-o{pitr}", f"{oitr}"], shell=True
                    )
                    if echo_dart:
                        dart_measurement_file(
                            os.path.basename(pitr), pitr, "png"
                        )
            except Exception as e:
                sys.stderr.write(f"{e}\n")


def dump_flamegraph(data, metric, file=None, echo_dart=False):
    """Dumps a flamegraph file"""
    from timemory.common import (
        popen,
        get_bin_script,
        dart_measurement_file,
    )

    _files = dump_entity(
        data, lambda x: x.to_flamegraph(metric), file, ".flamegraph.txt"
    )
    for itr in _files:
        flamegrapher = get_bin_script("flamegraph.pl")
        if itr is not None:
            if flamegrapher is None:
                if echo_dart is True:
                    # write_ctest_notes(itr)
                    dart_measurement_file(
                        os.path.basename(itr), itr, format="string", type="text"
                    )
            else:
                (retc, outs, errs) = popen(
                    [
                        flamegrapher,
                        "--hash",
                        "--inverted",
                        "--bgcolors",
                        "'#FFFFFF'",
                        itr,
                    ],
                    shell=True,
                )

                if outs is not None:
                    lbl = _get_label(itr)
                    sitr = _get_filename(itr, ".svg")
                    pitr = _get_filename(itr, ".png")

                    # write the SVG file
                    print(f"[{lbl}]|0> Outputting '{sitr}'...")
                    with open(sitr, "w") as fout:
                        fout.write(f"{outs}\n")

                    # generate png
                    pfile = _svg_to_png(pitr, svg_code=outs)

                    # echo svg and png
                    if echo_dart:
                        # write_ctest_notes(sitr)
                        dart_measurement_file(
                            os.path.basename(itr),
                            itr,
                            format="string",
                            type="text",
                        )
                        dart_measurement_file(
                            os.path.basename(sitr), sitr, "svg"
                        )
                        if pfile is not None:
                            dart_measurement_file(
                                os.path.basename(pitr), pitr, "png"
                            )
        else:
            pass


def dump_tabulate(dtype, data, metric, file=None, echo_dart=False):
    """Dumps a non-graphframe"""

    from tabulate import tabulate

    def _get_dataframe(x):
        import hatchet as ht

        return x.dataframe if isinstance(x, ht.graphframe.GraphFrame) else x

    _functors = {
        "html": lambda x: _get_dataframe(x).to_html(),
        "table": lambda x: tabulate(_get_dataframe(x), headers="keys"),
        "markdown": lambda x: _get_dataframe(x).to_markdown(),
        "markdown_grid": lambda x: _get_dataframe(x).to_markdown(
            tablefmt="grid"
        ),
    }

    _extensions = {
        "html": ".html",
        "table": ".txt",
        "markdown": ".md",
        "markdown_grid": ".md",
    }

    _files = dump_entity(data, _functors[dtype], file, _extensions[dtype])
    for itr in _files:
        if itr is not None and echo_dart is True:
            from timemory.common import dart_measurement_file

            dart_measurement_file(
                os.path.basename(itr), itr, format="string", type="text"
            )
            # write_ctest_notes(itr)


def dump_unknown(data, metric, file=None, echo_dart=False):
    """Dumps a non-graphframe"""
    _files = dump_entity(data, lambda x: x, file, ".txt")
    for itr in _files:
        if itr is not None and echo_dart is True:
            from timemory.common import dart_measurement_file

            dart_measurement_file(
                os.path.basename(itr), itr, format="string", type="text"
            )
            # write_ctest_notes(itr)


def dump(data, metric, dtype, file=None, echo_dart=False):
    # if isinstance(data, list):
    #    for itr in data:
    #        print(itr.dataframe)

    _success = False
    try:
        if dtype == "tree":
            dump_tree(data, metric, file, echo_dart)
        elif dtype == "dot":
            dump_dot(data, metric, file, echo_dart)
        elif dtype == "flamegraph":
            dump_flamegraph(data, metric, file, echo_dart)
        elif dtype in ("table", "markdown", "html", "markdown_grid"):
            dump_tabulate(dtype, data, metric, file, echo_dart)
        _success = True
    except (AttributeError, ImportError) as e:
        import sys

        sys.stderr.write(f"{e}\n")
        sys.stderr.flush()

    if not _success:
        dump_unknown(data, metric, file, echo_dart)
