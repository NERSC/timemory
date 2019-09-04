#!/usr/bin/env python

import sys
import argparse
# read in components and mangled enums
from timemory_types import components, mangled_enums


def generate_enum(component, indent):
    """
    This function generates a type trait label for C++
    """
    enumeration = mangled_enums.get(component, component)
    return "{}.value({}{}{}, {})".format(" " * indent, '"', enumeration, '"',
                                       enumeration.upper())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-V", "--verbose", help="Enable verbosity", default=0, type=int)
    parser.add_argument("-i", "--indent", help="Indentation", default=8, type=int)
    args = parser.parse_args()

    if args.verbose > 0:
        print("timemory components: [{}]\n".format(", ".join(components)))

    components.sort()
    outdata = "components_enum\n"
    for component in components:
        outdata += "{}\n".format(generate_enum(component, args.indent))

    outdata = outdata.strip("\n")
    outdata += ";"
    print("{}".format(outdata))
