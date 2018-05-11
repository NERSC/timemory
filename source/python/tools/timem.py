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

    parser.set_defaults(enable_handler=False)
    parser.set_defaults(quiet=False)

    args, left = parser.parse_known_args()
    # replace sys.argv with unknown args only
    sys.argv = sys.argv[:1]+left

    return args


#------------------------------------------------------------------------------#
#   Main execution
#
if __name__ == "__main__":

    try:
        args = handle_arguments()

        import timemory
        import subprocess as sp

        if len(sys.argv) > 1:
            # Python 2.x doesn't have "run"
            if sys.version_info[0] > 2:
                sp.run(sys.argv[1:])
            else:
                sp.call(sys.argv[1:])

        timemory_manager = timemory.manager()

        #----------------------------------------------------------------------#
        #   reporting to stdout
        #
        if not args.quiet:
            report = "{}".format(timemory_manager)
            if len(sys.argv) > 1:
                report = report.replace("[exe]", "[{}]".format(sys.argv[1]))
            print("\n{}".format(report))

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
