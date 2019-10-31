#!/usr/bin/env python

import sys
import argparse
# read in components and mangled enums
from timemory_types import components, mangled_enums


def generate_storage(component, key):
    """
    This function generates a type trait label for C++
    """
    return "TIMEMORY_{}_EXTERN_INIT({})".format(key.upper(), component)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-V", "--verbose",
                        help="Enable verbosity", default=0, type=int)
    parser.add_argument(
        "-i", "--indent", help="Indentation", default=4, type=int)
    args = parser.parse_args()

    if args.verbose > 0:
        print("timemory components: [{}]\n".format(", ".join(components)))

    components.sort()
    outdata = "\n"
    for component in components:
        outdata += "{}\n".format(generate_storage(component, "declare"))
    outdata += "\n"
    for component in components:
        outdata += "{}\n".format(generate_storage(component, "instantiate"))

    print("{}".format(outdata))
