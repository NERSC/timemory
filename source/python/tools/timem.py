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

        # record python overhead memory
        rss_init = timemory.rss_usage(record=True)

        # reset total timer to zero
        timemory_manager.reset_total_timer()

        # run the command
        if len(sys.argv) > 1:
            p = sp.Popen(sys.argv[1:])
            ret = p.wait()

        # stop the total timer to ensure no extra timer gets added
        timemory_manager.stop_total_timer()

        # subtract out python overhead memory
        timemory_manager -= rss_init

        #----------------------------------------------------------------------#
        #   reporting to stdout
        #
        if not args.quiet:

            fmt = ": %w wall, %u user + %s system = %t cpu (%p%) [%T], %M peak rss [%A]"
            # fix the format
            if sys.version_info[0] > 2:
                timemory.format.timer.set_default_format(fmt)
            else:
                timer_format = timemory.format.timer()
                timer_format.set_default_format(fmt)

            # update the format
            timemory_manager.update_total_timer_format()

            # generate report
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

    sys.exit(ret)
