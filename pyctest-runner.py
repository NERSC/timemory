#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyCTest driver for TiMemory
"""

import os, sys, platform, traceback, warnings, shutil
import multiprocessing as mp
import pyctest.pyctest as pyctest
import pyctest.helpers as helpers


#------------------------------------------------------------------------------#
def configure():

    # Get pyctest argument parser that include PyCTest arguments
    parser = helpers.ArgumentParser(project_name="TiMemory",
                                    source_dir=os.getcwd(),
                                    binary_dir=os.path.join(
                                        os.getcwd(), "build-timemory"),
                                    build_type="Release",
                                    vcs_type="git",
                                    use_launchers=False)

    parser.add_argument("--arch", help="TIMEMORY_USE_ARCH=ON",
                        default=False, action='store_true')
    parser.add_argument("--gperf", help="TIMEMORY_USE_GPERF=ON",
                        default=False, action='store_true')
    parser.add_argument("--sanitizer", help="TIMEMORY_USE_SANITIZER=ON",
                        default=False, action='store_true')
    parser.add_argument("--coverage", help="TIMEMORY_USE_COVERAGE=ON",
                        default=False, action='store_true')
    parser.add_argument("--static-analysis", help="TIMEMORY_USE_CLANG_TIDY=ON",
                        default=False, action='store_true')
    parser.add_argument("--no-papi", help="TIMEMORY_USE_PAPI=OFF",
                        default=False, action='store_true')
    parser.add_argument("--no-mpi", help="TIMEMORY_USE_MPI=OFF",
                        default=False, action='store_true')
    parser.add_argument("--no-py", help="TIMEMORY_BUILD_PYTHON=OFF",
                        default=False, action='store_true')
    parser.add_argument("--no-c", help="TIMEMORY_BUILD_C=OFF",
                        default=False, action='store_true')

    args = parser.parse_args()

    if os.environ.get("CTEST_SITE") is not None:
        pyctest.set("CTEST_SITE", "{}".format(os.environ.get("CTEST_SITE")))

    if os.path.exists(os.path.join(pyctest.BINARY_DIRECTORY, "CMakeCache.txt")):
        from pyctest import cmake_executable as cm
        from pyctest import version_info as _pyctest_version
        if (_pyctest_version[0] == 0 and
                _pyctest_version[1] == 0 and
                _pyctest_version[2] < 11):
            cmd = pyctest.command(
                [cm, "--build", pyctest.BINARY_DIRECTORY, "--target", "clean"])
            cmd.SetWorkingDirectory(pyctest.BINARY_DIRECTORY)
            cmd.SetOutputQuiet(True)
            cmd.SetErrorQuiet(True)
            cmd.Execute()
        else:
            from pyctest.cmake import CMake
            CMake("--build", pyctest.BINARY_DIRECTORY, "--target", "clean")
        helpers.RemovePath(os.path.join(pyctest.BINARY_DIRECTORY, "CMakeCache.txt"))

    if args.gperf:
        pyctest.copy_files(["gperf_cpu_profile.sh", "gperf_heap_profile.sh"],
                           os.path.join(pyctest.SOURCE_DIRECTORY, ".scripts"),
                           pyctest.BINARY_DIRECTORY)

    return args


#------------------------------------------------------------------------------#
#
def run_pyctest():

    #--------------------------------------------------------------------------#
    # run argparse, checkout source, copy over files
    #
    args = configure()

    #--------------------------------------------------------------------------#
    # Compiler version
    #
    if os.environ.get("CXX") is None:
        os.environ["CXX"] = helpers.FindExePath("c++")
    cmd = pyctest.command([os.environ["CXX"], "-dumpversion"])
    cmd.SetOutputStripTrailingWhitespace(True)
    cmd.Execute()
    compiler_version = cmd.Output()

    #--------------------------------------------------------------------------#
    # Set the build name
    #
    pyctest.BUILD_NAME = "{} {} {} {} {} {}".format(
        pyctest.GetGitBranch(pyctest.SOURCE_DIRECTORY),
        platform.uname()[0],
        helpers.GetSystemVersionInfo(),
        platform.uname()[4],
        os.path.basename(os.path.realpath(os.environ["CXX"])),
        compiler_version)
    pyctest.BUILD_NAME = '-'.join(pyctest.BUILD_NAME.split())

    #--------------------------------------------------------------------------#
    #   build specifications
    #
    build_opts = {
        "TIMEMORY_BUILD_PYTHON": "ON",
        "TIMEMORY_BUILD_C": "ON",
        "TIMEMORY_USE_MPI": "ON",
        "TIMEMORY_USE_PAPI": "ON",
        "TIMEMORY_USE_ARCH": "OFF",
        "TIMEMORY_USE_GPERF": "OFF",
        "TIMEMORY_USE_SANITIZER": "OFF",
        "TIMEMORY_USE_COVERAGE" : "OFF",
        "TIMEMORY_USE_CLANG_TIDY": "OFF",
    }

    if args.no_c:
        build_opts["TIMEMORY_BUILD_C"] = "OFF"
    else:
        pyctest.BUILD_NAME = "{} C".format(pyctest.BUILD_NAME)
    if args.no_py:
        build_opts["TIMEMORY_BUILD_PYTHON"] = "OFF"
    else:
        pyctest.BUILD_NAME = "{} PY".format(pyctest.BUILD_NAME)
    if args.no_mpi:
        build_opts["TIMEMORY_USE_MPI"] = "OFF"
    else:
        pyctest.BUILD_NAME = "{} MPI".format(pyctest.BUILD_NAME)
    if args.no_papi:
        build_opts["TIMEMORY_USE_PAPI"] = "OFF"
    else:
        pyctest.BUILD_NAME = "{} PAPI".format(pyctest.BUILD_NAME)
    if args.arch:
        pyctest.BUILD_NAME = "{} arch".format(pyctest.BUILD_NAME)
        build_opts["TIMEMORY_USE_ARCH"] = "ON"
    if args.gperf:
        pyctest.BUILD_NAME = "{} gperf".format(pyctest.BUILD_NAME)
        build_opts["TIMEMORY_USE_GPERF"] = "ON"
        warnings.warn(
            "Forcing build type to 'RelWithDebInfo' when gperf is enabled")
        pyctest.BUILD_TYPE = "RelWithDebInfo"
    if args.sanitizer:
        pyctest.BUILD_NAME = "{} asan".format(pyctest.BUILD_NAME)
        build_opts["TIMEMORY_USE_SANITIZER"] = "ON"
    if args.static_analysis:
        build_opts["TIMEMORY_USE_CLANG_TIDY"] = "ON"
    if args.coverage:
        gcov_exe = helpers.FindExePath("gcov")
        if gcov_exe is not None:
            pyctest.COVERAGE_COMMAND = "{}".format(gcov_exe)
            build_opts["TIMEMORY_USE_COVERAGE"] = "ON"
            warnings.warn(
                "Forcing build type to 'Debug' when coverage is enabled")
            pyctest.BUILD_TYPE = "Debug"

    # split and join with dashes
    pyctest.BUILD_NAME = '-'.join(pyctest.BUILD_NAME.replace('/', '-').split())

    # default options
    cmake_args = "-DCMAKE_BUILD_TYPE={} -DTIMEMORY_BUILD_EXAMPLES=ON".format(
        pyctest.BUILD_TYPE)

    # customized from args
    for key, val in build_opts.items():
        cmake_args = "{} -D{}={}".format(cmake_args, key, val)

    #--------------------------------------------------------------------------#
    # how to build the code
    #
    ctest_cmake_cmd = "${CTEST_CMAKE_COMMAND}"
    pyctest.CONFIGURE_COMMAND = "{} {} {}".format(
        ctest_cmake_cmd, cmake_args, pyctest.SOURCE_DIRECTORY)

    #--------------------------------------------------------------------------#
    # how to build the code
    #
    pyctest.BUILD_COMMAND = "{} --build {} --target all".format(
        ctest_cmake_cmd, pyctest.BINARY_DIRECTORY)

    #--------------------------------------------------------------------------#
    # parallel build
    #
    if platform.system() != "Windows":
        pyctest.BUILD_COMMAND = "{} -- -j{} VERBOSE=1".format(
            pyctest.BUILD_COMMAND, mp.cpu_count())
    else:
        pyctest.BUILD_COMMAND = "{} -- /MP -A x64".format(pyctest.BUILD_COMMAND)


    #--------------------------------------------------------------------------#
    # how to update the code
    #
    git_exe = helpers.FindExePath("git")
    pyctest.UPDATE_COMMAND = "{}".format(git_exe)
    pyctest.set("CTEST_UPDATE_TYPE", "git")
    pyctest.set("CTEST_GIT_COMMAND", "{}".format(git_exe))

    #--------------------------------------------------------------------------#
    # find the CTEST_TOKEN_FILE
    #
    if args.pyctest_token_file is None and args.pyctest_token is None:
        home = helpers.GetHomePath()
        if home is not None:
            token_path = os.path.join(
                home, os.path.join(".tokens", "nersc-cdash"))
            if os.path.exists(token_path):
                pyctest.set("CTEST_TOKEN_FILE", token_path)

    #--------------------------------------------------------------------------#
    # construct a command
    #
    def construct_command(cmd, args, clobber=False):
        _cmd = []
        if args.gperf:
            _cmd.append(os.path.join(pyctest.BINARY_DIRECTORY,
                                     "gperf-cpu-profile.sh"))
            pyctest.add_note(pyctest.BINARY_DIRECTORY,
                             "gperf.cpu.prof.{}.0.txt".format(
                                 os.path.basename(cmd[0])),
                             clobber=clobber)
            pyctest.add_note(pyctest.BINARY_DIRECTORY,
                             "gperf.cpu.prof.{}.0.cum.txt".format(
                                 os.path.basename(cmd[0])),
                             clobber=False)
        else:
            _cmd.append("./timem")
        _cmd.extend(cmd)
        return _cmd

    #--------------------------------------------------------------------------#
    # standard environment settings for tests, adds profile to notes
    #
    def test_env_settings(prof_fname, clobber=False, extra=""):
        return "NUM_THREADS={};{}".format(
            mp.cpu_count(), extra)

    #pyctest.set("ENV{GCOV_PREFIX}", pyctest.BINARY_DIRECTORY)
    #pyctest.set("ENV{GCOV_PREFIX_STRIP}", "4")

    #--------------------------------------------------------------------------#
    # create tests
    #
    if not args.no_c:
        pyctest.test("test_c_timing", construct_command(["./test_c_timing"], args, clobber=True),
                     {"WORKING_DIRECTORY": pyctest.BINARY_DIRECTORY, "LABELS": pyctest.PROJECT_NAME})
    pyctest.test("test_cxx_overhead", construct_command(["./test_cxx_overhead"], args),
                 {"WORKING_DIRECTORY": pyctest.BINARY_DIRECTORY, "LABELS": pyctest.PROJECT_NAME})
    pyctest.test("test_cxx_tuple", construct_command(["./test_cxx_tuple"], args, clobber=True),
                 {"WORKING_DIRECTORY" : pyctest.BINARY_DIRECTORY, "LABELS": pyctest.PROJECT_NAME})
    #pyctest.test("test_cxx_total", construct_command(["./test_cxx_total"], args),
    #             {"WORKING_DIRECTORY": pyctest.BINARY_DIRECTORY, "LABELS": pyctest.PROJECT_NAME})
    #pyctest.test("test_cxx_timing", construct_command(["./test_cxx_timing"], args, clobber=True),
    #             {"WORKING_DIRECTORY": pyctest.BINARY_DIRECTORY, "LABELS": pyctest.PROJECT_NAME})
    #if args.mpi:
    #    pyctest.test("test_cxx_mpi_timing", construct_command(["./test_cxx_mpi_timing"], args, clobber=True),
    #                 {"WORKING_DIRECTORY": pyctest.BINARY_DIRECTORY, "LABELS": pyctest.PROJECT_NAME})

    pyctest.generate_config(pyctest.BINARY_DIRECTORY)
    pyctest.generate_test_file(os.path.join(pyctest.BINARY_DIRECTORY, "examples"))
    pyctest.run(pyctest.ARGUMENTS, pyctest.BINARY_DIRECTORY)


#------------------------------------------------------------------------------#
if __name__ == "__main__":

    try:

        run_pyctest()

    except Exception as e:
        print('Error running pyctest - {}'.format(e))
        exc_type, exc_value, exc_trback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_trback, limit=10)
        sys.exit(1)

    sys.exit(0)
