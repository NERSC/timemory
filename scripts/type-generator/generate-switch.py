#!/usr/bin/env python

import sys
import argparse
# read in components and mangled enums
from timemory_types import components, mangled_enums


def generate_case_label(component, func, indent_tabs=3, spaces=4, reference=True, template=True):
    """
    This function generates a case label for C++
    """
    enumeration = mangled_enums.get(component, component)

    spacer = " "*spaces             # a spacer of spaces length
    atab = "{}".format(spacer)      # the generic tab
    ftab = atab*indent_tabs         # the first tab level

    init_str = "." if reference else "->"
    if template:
        init_str += "template "
    return "{2}case {1}: obj{3}{4}<{0}>(); break;\n".format(
        component, enumeration.upper(), ftab, init_str, func)


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
    parser.add_argument("-f", "--function",
                        help="Function name", default="init", type=str)

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
    outdata = ""
    for component in components:
        outdata += "{}".format(generate_case_label(component, args.function,
                                                   args.tabs_per_indent, args.spaces_per_tab,
                                                   not args.pointer, not args.specialized))
    outdata = outdata.strip("\n")
    spacer = " "*args.spaces_per_tab    # a spacer of spaces length
    atab = "{}".format(spacer)          # the generic tab
    ftab = atab*args.tabs_per_indent    # the first tab level

    outdata += "\n{}case TIMEMORY_COMPONENTS_END:\n{}default: break;".format(
        ftab, ftab)

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
