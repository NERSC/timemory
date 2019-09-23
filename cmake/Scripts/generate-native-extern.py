#!/usr/bin/env python

import sys
import argparse
# read in components and mangled enums
from timemory_types import native_components


def generate_extern(component, key, alias, suffix, specify_comp=True):
    """
    This function generates a type trait label for C++
    """
    if specify_comp:
        return "TIMEMORY_{}_EXTERN_{}({}, {})".format(
            key.upper(), suffix.upper(), alias, "::tim::component::{}".format(component))
    else:
        return "TIMEMORY_{}_EXTERN_{}({})".format(
            key.upper(), suffix.upper(), alias)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-V", "--verbose",
                        help="Enable verbosity", default=0, type=int)
    parser.add_argument(
        "-i", "--indent", help="Indentation", default=4, type=int)
    args = parser.parse_args()

    if args.verbose > 0:
        print("timemory components: [{}]\n".format(
            ", ".join(native_components)))

    native_components.sort()
    outdata = "\n"
    for key in ["declare", "instantiate"]:
        for suffix in ["tuple", "list", "hybrid"]:
            specify_comp = False if suffix == "hybrid" else True
            for component in native_components:
                outdata += "{}\n".format(generate_extern(
                    component, key, "_".join([component, "t"]),
                    suffix, specify_comp))
            outdata += "\n"

    print("{}".format(outdata))
