#!/usr/bin/env python

import sys
import argparse
# read in components and mangled enums
from timemory_types import components


def generate_latex(component, indent):
    """
    This function generates a type trait label for C++
    """
    return "{}\item \lstinlinec{}{}{}".format(" " * indent, "{", component, "}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-V", "--verbose", help="Enable verbosity", default=0, type=int)
    parser.add_argument("-i", "--indent", help="Indentation", default=4, type=int)
    args = parser.parse_args()

    if args.verbose > 0:
        print("timemory components: [{}]\n".format(", ".join(components)))

    components.sort()
    outdata = "\\begin{itemize}\n"
    for component in components:
        outdata += "{}\n".format(generate_latex(component, args.indent))

    outdata += "\\end{itemize}"
    print("{}".format(outdata))
