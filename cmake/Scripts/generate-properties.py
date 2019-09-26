#!/usr/bin/env python

import sys
import argparse
# read in components and mangled enums
from timemory_types import components, mangled_enums


def generate_get_enum(component, indent_tabs=3, spaces=4, reference=True, template=True):
    """
    This function generates a case label for C++
    """
    enumeration = mangled_enums.get(component, component)

    spacer = " "*spaces             # a spacer of spaces length
    atab = "{}".format(spacer)      # the generic tab
    ftab = atab*indent_tabs         # the first tab level

    _beg = "template <>\nstruct get_enum<{}>\n{}".format(
        component, "{")
    _comp = "\n{}static constexpr TIMEMORY_COMPONENT value = {};".format(
        ftab, enumeration.upper())
    _bool = "\n{}static bool& has_storage() {} static thread_local bool _instance = false; return _instance; {}".format(
        ftab, "{", "}")
    _end = "\n{};".format("}")
    return _beg + _comp + _bool + _end


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
        outdata += "{}\n".format(generate_get_enum(component,
                                                   args.tabs_per_indent, args.spaces_per_tab))
        outdata += "\n//{}//\n\n".format("-" * 86)
    outdata = outdata.strip("\n")
    print("{}".format(outdata))
