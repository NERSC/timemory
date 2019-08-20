#!/usr/bin/env python

import sys
import argparse
# read in components and mangled enums
from timemory_types import components, mangled_enums, mangled_strings


def generate_if_statement(component, idx, indent_tabs=2, spaces=4, vec_var="vec"):
    """
    This function generates a case label for C++
    """
    enumeration = mangled_enums.get(component, component)
    strcomponent = mangled_strings.get(component, [component])
    if component not in strcomponent:
        strcomponent += [component]
    strcomponent.sort()

    checkstatement = ""
    for i in range(0, len(strcomponent)):
        checkstatement += "itr == \"{}\"".format(strcomponent[i])
        if i + 1 < len(strcomponent):
            checkstatement += " || "

    tab = " " * spaces * indent_tabs
    cond = "if" if idx == 0 else "else if"
    return "{}{}({}) {} {}.push_back({}); {}".format(
        tab, cond, checkstatement, "{", vec_var, enumeration.upper(), "}")


if __name__ == "__main__":

    rbegin = "// GENERATE_SWITCH_REPLACE_BEGIN"
    rend = "// GENERATE_SWITCH_REPLACE_END"

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--additional",
                        help="Additional components to produce case labels for", nargs='*')
    parser.add_argument("-i", "--input", help="input file for replacing",
                        type=str, default=None)
    parser.add_argument("-o", "--output", help="output components from file",
                        type=str, default=None)
    parser.add_argument("-e", "--exclude",
                        help="exclude components", action='store_true')
    parser.add_argument("-s", "--spaces-per-tab",
                        help="Number of spaces in a tab", type=int, default=4)
    parser.add_argument("-t", "--tabs-per-indent",
                        help="Number of tabs to indent", type=int, default=2)
    parser.add_argument("-r", "--replace", action='store_true',
                        help="Replace area between '{}' and '{}' in the output file".format(rbegin, rend))
    parser.add_argument("-c", "--config", help="alternative to -i/--input when replacing",
                        type=str, default=None)
    parser.add_argument("-v", "--vec-var", help="Name of the enumeration vector variable",
                        type=str, default="vec")
    parser.add_argument("-I", "--iter-var", help="Name of the iteration variable",
                        type=str, default="itr")
    parser.add_argument("-V", "--verbose", help="Enable verbosity", default=0, type=int)

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

    component_options = []
    for component in components:
        component_options += mangled_strings.get(component, [component])
        if component not in component_options:
            component_options.append(component)
    component_options.sort()

    outdata = ""
    idx = 0
    for component in components:
        outdata += "{}\n".format(generate_if_statement(component, idx,
                                                       args.tabs_per_indent, args.spaces_per_tab,
                                                       args.vec_var))
        idx += 1
    tab = " " * args.spaces_per_tab * args.tabs_per_indent
    outdata += '{}else {} fprintf(stderr, "Unknown component label: %s{}", {}.c_str()); {}'.format(
        tab, "{", ". Valid choices are: {}\\n".format(component_options), args.iter_var, "}")

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
