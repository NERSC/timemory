#!/usr/bin/env python

import sys
import argparse
# read in components and mangled enums
from timemory_types import components, mangled_enums


def generate_enum(component, ncount, indent):
    """
    This function generates a type trait label for C++
    """
    enumeration = mangled_enums.get(component, component)
    return "{}{:30} = {},".format(" " * indent, enumeration.upper(), ncount)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-V", "--verbose", help="Enable verbosity", default=0, type=int)
    parser.add_argument("-i", "--indent", help="Indentation", default=4, type=int)
    args = parser.parse_args()

    if args.verbose > 0:
        print("timemory components: [{}]\n".format(", ".join(components)))

    components.sort()
    outdata = "enum TIMEMORY_NATIVE_COMPONENT\n{\n"
    for i in range(0, len(components)):
        outdata += "{}\n".format(generate_enum(components[i], i, args.indent))

    outdata += "{}{} {} = \n{}    ({} + {})\n".format(" " * args.indent,
                                       "TIMEMORY_USER_COMPONENT_ENUM",
                                       "TIMEMORY_COMPONENTS_END", " " * args.indent,
                                       "TIMEMORY_COMPONENT_ENUM_SIZE",
                                       "TIMEMORY_USER_COMPONENT_ENUM_SIZE")
    outdata += "};"
    print("{}".format(outdata))
