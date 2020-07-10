#!/usr/bin/env python

import sys
import argparse
# read in components and mangled enums
from timemory_types import traits


def generate_trait(trait, component, inherit):
    """
    This function generates a type trait label for C++
    """
    return "template <>\nstruct {}<component::{}> : {} {};\n".format(
        trait, component, inherit, "{}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-V", "--verbose",
                        help="Enable verbosity", default=0, type=int)
    parser.add_argument("-w", "--page-width",
                        help="Page width", default=90, type=int)
    args = parser.parse_args()

    dashes = "-" * (args.page_width - 4)
    outdata = ""
    for trait, spec in traits.items():
        inherit = spec[0]
        components = spec[1]
        outdata += "//{0}//\n//\n//\t\t\t{1}\n//\n//{0}//\n\n".format(
            dashes, trait.replace("_", " ").upper())
        if args.verbose > 0:
            print("timemory components: [{}]\n".format(", ".join(components)))
        for component in components:
            outdata += "{}\n".format(generate_trait(trait, component, inherit))

    outdata = outdata.strip("\n")
    print("{}".format(outdata))
