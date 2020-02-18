#!/usr/bin/env python

import sys
import argparse
# read in components and mangled enums
from timemory_types import components, mangled_enums, mangled_strings


def generate_get_enum(component):
    """
    This function generates a case label for C++
    """
    enumeration = mangled_enums.get(component, component).upper()
    strings = ["{}".format(component)] + mangled_strings.get(component, [])
    if len(strings) == 1:
        strings += ['']

    quoted = ('"{}"'.format(x) for x in strings)

    return "TIMEMORY_PROPERTY_SPECIALIZATION({}, {}, {})".format(
           component, enumeration, ", ".join(quoted))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--additional", nargs='*',
                        help="Additional components to produce case labels for")
    parser.add_argument("-e", "--exclude",
                        help="exclude components", action='store_true')
    parser.add_argument("-s", "--spaces-per-tab",
                        help="Number of spaces in a tab", type=int, default=4)
    parser.add_argument("-t", "--tabs-per-indent",
                        help="Number of tabs to indent", type=int, default=1)
    parser.add_argument("-V", "--verbose",
                        help="Enable verbosity", default=0, type=int)

    args = parser.parse_args()

    if args.additional:
        components += args.additional

    components.sort()
    if args.verbose > 0:
        print("timemory components: [{}]\n".format(", ".join(components)))
    outdata = ""
    for component in components:
        outdata += "{}\n".format(generate_get_enum(component))
        outdata += "\n//{}//\n\n".format("-" * 86)
    outdata = outdata.strip("\n")
    print("{}".format(outdata))
