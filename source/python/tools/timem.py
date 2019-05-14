#!@PYTHON_EXECUTABLE@

import sys
import traceback
import argparse


"""
A) Run a process and report TiMemory's timing and memory resuls

    Usage:
        timem ls -l

B) Run a process and pass a JSON serialization to a custom
   'timemory_json_handler.py' script so that the data can be processed

    Usage:
        timem --enable-json-handler --timem-quiet sleep 3
"""


#------------------------------------------------------------------------------#
#   Allow some options
#
def handle_arguments():
    """
    Processes options for timem and removes known
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-json-handler',
        required=False, action='store_true', dest='enable_handler',
        help="Enable:\n\t{}\n\t{} to handle timing info".format(
            "import timemory_json_handler",
            "timemory_json_handler.receive(timemory.json())"))
    parser.add_argument('--timem-quiet',
        required=False, action='store_true', dest='quiet',
        help="Suppress reporting to stdout")
    parser.add_argument('--debug',
        required=False, action='store_true', dest='debug',
        help="Enable debug output")

    parser.set_defaults(enable_handler=False)
    parser.set_defaults(quiet=False)
    parser.set_defaults(debug=False)

    args, left = parser.parse_known_args()
    # replace sys.argv with unknown args only
    sys.argv = sys.argv[:1]+left

    return args


#------------------------------------------------------------------------------#
#   Main execution
#
if __name__ == "__main__":

    ret = 0
    try:
        #----------------------------------------------------------------------#
        #   parse arguments
        #
        args = handle_arguments()

        #----------------------------------------------------------------------#
        #   import and setup
        #
        import timemory
        import subprocess as sp

        # grab a manager handle
        timemory_manager = timemory.manager()

        # rusage records child processes
        timemory.set_rusage_children()

        components = []
        for c in ["wall_clock", "sys_clock", "cpu_clock", "cpu_util", "peak_rss",
                  "num_minor_page_faults", "num_major_page_faults",
                  "voluntary_context_switch", "priority_context_switch"]:
            components.append(getattr(timemory.component, c))

        # run the command
        exe_name = "[pytimem]"
        if len(sys.argv) > 1:
            exe_name = "[{}]".format(sys.argv[1])

        comp = timemory.component_tuple(components, exe_name)
        comp.start()
        if len(sys.argv) > 1:
            p = sp.Popen(sys.argv[1:])
            ret = p.wait()
        comp.stop()

        print("{}".format(comp))

        # generate report
        timemory.report();

        #----------------------------------------------------------------------#
        #   handler
        #
        if args.enable_handler:

            try:
                import timemory_json_handler
                timemory_json_handler.receive(sys.argv[1:], timemory_manager.json())

            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback)

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)

    sys.exit(ret)
