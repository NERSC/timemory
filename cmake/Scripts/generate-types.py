#!/usr/bin/env python

import sys
import argparse
# read in components and mangled enums
from timemory_types import components, mangled_enums, recommended_types


def generate_list(component, indent_tabs=3, spaces=4, reference=True, template=True):
    """
    This function generates a case label for C++
    """
    enumeration = mangled_enums.get(component, component)

    spacer = " "*spaces           # a spacer of spaces length
    atab = "{}".format(spacer)  # the generic tab
    ftab = atab*indent_tabs     # the first tab level
    stab = atab*(indent_tabs+1)  # the second tab level

    init_str = "." if reference else "->"
    if template:
        init_str += "template "
    return "{2}case {1}: obj{3}init<{0}>(); break;\n".format(
        component, enumeration.upper(), ftab, init_str)


if __name__ == "__main__":

    rbegin = "// GENERATE_SWITCH_REPLACE_BEGIN"
    rend = "// GENERATE_SWITCH_REPLACE_END"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--additional", help="Additional components to produce case labels for", nargs='*')
    parser.add_argument(
        "-i", "--input", help="input file for replacing", type=str, default=None)
    parser.add_argument(
        "-o", "--output", help="output components from file", type=str, default=None)
    parser.add_argument("-e", "--exclude",
                        help="exclude components", action='store_true')
    parser.add_argument("-s", "--spaces-per-tab",
                        help="Number of spaces in a tab", type=int, default=4)
    parser.add_argument("-t", "--tabs-per-indent",
                        help="Number of tabs to indent", type=int, default=3)
    parser.add_argument("-r", "--replace", action='store_true',
                        help="Replace area between '{}' and '{}' in the output file".format(rbegin, rend))
    parser.add_argument(
        "-c", "--config", help="alternative to -i/--input when replacing", type=str, default=None)
    parser.add_argument("-P", "--pointer", help="Object operating on is pointer ('->') instead of reference ('.')",
                        action='store_true')
    parser.add_argument("-S", "--specialized", help="Initialization is specialized (not a template)",
                        action='store_true')
    parser.add_argument("-V", "--verbose",
                        help="Enable verbosity", default=0, type=int)

    args = parser.parse_args()

    if args.replace and args.input is None and args.output is None:
        raise Exception(
            "-r/--replace requires specifying -o/--output or -i/--input")

    # substitution string for replacing
    subdata = None

    if args.input is not None:
        with open(args.input, 'r') as f:
            subdata = f.read()

    if args.config is not None:
        with open(args.config, 'r') as f:
            components = f.read().strip().split()
        print("timemory components: [{}]\n".format(
            ", ".join(components)))

    if args.additional:
        components += args.additional

    if subdata is None and args.replace:
        if args.input is None:
            print("reading '{}' for replacement...".format(args.output))
            with open(args.output, 'r') as f:
                subdata = f.read()

    components.sort()
    if args.verbose > 0:
        print("timemory components: [{}]\n".format(", ".join(components)))

    #
    outdata = "using complete_tuple_t = std::tuple<\n"
    for component in components:
        outdata += "\tcomponent::{},\n".format(component)
    outdata = outdata.strip("\n")
    outdata = outdata.strip(",")
    outdata += ">;\n\n"

    outdata += "using complete_auto_list_t = auto_list<\n"
    for component in components:
        outdata += "\tcomponent::{},\n".format(component)
    outdata = outdata.strip("\n")
    outdata = outdata.strip(",")
    outdata += ">;\n\n"
    outdata += "using complete_list_t = complete_auto_list_t::component_type;\n\n"

    a_hybdata = "using recommended_auto_hybrid_t = auto_hybrid<\n"
    c_hybdata = "using recommended_hybrid_t = component_hybrid<\n"
    for key, items in recommended_types.items():
        outdata += "using recommended_auto_{0}_t = auto_{0}<\n".format(key)
        for item in items:
            outdata += "\tcomponent::{},\n".format(item)
        outdata = outdata.strip("\n")
        outdata = outdata.strip(",")
        outdata += ">;\n\n"
        outdata += "using recommended_{0}_t = recommended_auto_{0}_t::component_type;\n\n".format(
            key)
        a_hybdata += "\trecommended_{0}_t,\n".format(key)
        c_hybdata += "\trecommended_{0}_t,\n".format(key)
    a_hybdata = a_hybdata.strip("\n")
    a_hybdata = a_hybdata.strip(",")
    a_hybdata += ">;\n\n"
    c_hybdata = c_hybdata.strip("\n")
    c_hybdata = c_hybdata.strip(",")
    c_hybdata += ">;\n\n"

    outdata += a_hybdata
    outdata += c_hybdata

    if subdata is not None:
        try:
            sbegin = subdata.find(rbegin) + len(rbegin) + 1
            send = subdata.find(rend)
            substr = subdata[sbegin:send]
            outdata += " " * args.spaces_per_tab * (args.tabs_per_indent + 1)
            if args.verbose > 1:
                print("sbegin = {}\nsend = {}\nsubstring:\n{}\n\noutdata:\n{}\n".format(
                      sbegin, send, substr, outdata))
            outdata = subdata.replace(substr, outdata)
            if args.verbose > 1:
                print("converted:\n{}\n".format(subdata))
        except Exception as e:
            print(e)
            raise

    if args.output is not None:
        with open(args.output, 'w') as f:
            f.write(outdata)
    else:
        print("{}".format(outdata))
