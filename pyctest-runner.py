#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyCTest driver for TiMemory
"""

import os
import sys
import shutil
import platform
import traceback
import warnings
import multiprocessing as mp
import pyctest.pyctest as pyctest
import pyctest.helpers as helpers

clobber_notes = True


#------------------------------------------------------------------------------#
def get_branch(wd=pyctest.SOURCE_DIRECTORY):
    cmd = pyctest.command(["git", "show", "-s", "--pretty=%d", "HEAD"])
    cmd.SetOutputStripTrailingWhitespace(True)
    cmd.SetWorkingDirectory(wd)
    cmd.Execute()
    branch = cmd.Output()
    branch = branch.split(" ")
    if branch:
        branch = branch[len(branch)-1]
        branch = branch.strip(")")
    if not branch:
        branch = pyctest.GetGitBranch(wd)
    return branch


#------------------------------------------------------------------------------#
def configure():

    # Get pyctest argument parser that include PyCTest arguments
    parser = helpers.ArgumentParser(project_name="TiMemory",
                                    source_dir=os.getcwd(),
                                    binary_dir=os.path.join(
                                        os.getcwd(), "build-timemory", platform.system()),
                                    build_type="Release",
                                    vcs_type="git",
                                    use_launchers=False)

    parser.add_argument("--arch", help="TIMEMORY_USE_ARCH=ON",
                        default=False, action='store_true')
    parser.add_argument("--profile", help="Run gperf profiler",
                        default=None, type=str, choices=("cpu", "heap"))
    parser.add_argument("--sanitizer", help="Type of sanitizer",
                        default=None, type=str, choices=("leak", "memory", "address", "thread"))
    parser.add_argument("--coverage", help="TIMEMORY_USE_COVERAGE=ON",
                        default=False, action='store_true')
    parser.add_argument("--static-analysis", help="TIMEMORY_USE_CLANG_TIDY=ON",
                        default=False, action='store_true')
    parser.add_argument("--cuda", help="TIMEMORY_USE_CUDA=ON",
                        default=False, action='store_true')
    parser.add_argument("--no-papi", help="TIMEMORY_USE_PAPI=OFF",
                        default=False, action='store_true')
    parser.add_argument("--no-mpi", help="TIMEMORY_USE_MPI=OFF",
                        default=False, action='store_true')
    parser.add_argument("--no-py", help="TIMEMORY_BUILD_PYTHON=OFF",
                        default=False, action='store_true')
    parser.add_argument("--no-c", help="TIMEMORY_BUILD_C=OFF",
                        default=False, action='store_true')
    parser.add_argument("--no-gtest", help="TIMEMORY_BUILD_GTEST=OFF",
                        default=False, action='store_true')
    parser.add_argument("--extra-optimizations",
                        help="TIMEMORY_BUILD_EXTRA_OPTIMIZATIONS=ON",
                        default=False, action='store_true')
    parser.add_argument("--no-extern-templates", help="TIMEMORY_BUILD_EXTERN_TEMPLATES=OFF",
                        default=False, action='store_true')
    # parser.add_argument("--build-libs", help="Build library type(s)", default=["shared,static"],
    #                     nargs='*', type=str, choices=["static", "shared"])

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
        helpers.RemovePath(os.path.join(
            pyctest.BINARY_DIRECTORY, "CMakeCache.txt"))

    if platform.system() != "Linux":
        args.no_papi = True

    # always echo dart measurements
    os.environ["TIMEMORY_DART_OUTPUT"] = "ON"

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
        get_branch(pyctest.SOURCE_DIRECTORY),
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
        "TIMEMORY_BUILD_C": "ON",
        "TIMEMORY_BUILD_GTEST": "ON",
        "TIMEMORY_BUILD_PYTHON": "ON",
        "TIMEMORY_BUILD_EXTERN_TEMPLATES": "ON",
        "TIMEMORY_BUILD_EXTRA_OPTIMIZATIONS": "OFF",
        "TIMEMORY_USE_MPI": "ON",
        "TIMEMORY_USE_PAPI": "ON",
        "TIMEMORY_USE_ARCH": "OFF",
        "TIMEMORY_USE_CUDA": "OFF",
        "TIMEMORY_USE_GPERF": "OFF",
        "TIMEMORY_USE_SANITIZER": "OFF",
        "TIMEMORY_USE_COVERAGE": "OFF",
        "TIMEMORY_USE_CLANG_TIDY": "OFF",
    }

    # if "shared" in args.build_libs:
    #     build_opts["BUILD_SHARED_LIBS"] = "ON"
    # else:
    #     build_opts["BUILD_SHARED_LIBS"] = "OFF"

    # if "static" in args.build_libs:
    #     build_opts["BUILD_STATIC_LIBS"] = "ON"
    # else:
    #     build_opts["BUILD_STATIC_LIBS"] = "OFF"

    if args.no_extern_templates:
        build_opts["TIMEMORY_BUILD_EXTERN_TEMPLATES"] = "OFF"
        build_opts["USE_EXTERN_TEMPLATES"] = "OFF"
    else:
        build_opts["USE_EXTERN_TEMPLATES"] = "ON"

    if args.no_gtest:
        build_opts["TIMEMORY_BUILD_GTEST"] = "OFF"

    if args.no_c:
        build_opts["TIMEMORY_BUILD_C"] = "OFF"
    else:
        pyctest.BUILD_NAME = "{} C".format(pyctest.BUILD_NAME)

    if args.no_py:
        build_opts["TIMEMORY_BUILD_PYTHON"] = "OFF"
    else:
        pyver = "{}.{}.{}".format(
            sys.version_info[0], sys.version_info[1], sys.version_info[2])
        pyctest.BUILD_NAME = "{} PY-{}".format(pyctest.BUILD_NAME, pyver)

    if args.extra_optimizations:
        build_opts["TIMEMORY_BUILD_EXTRA_OPTIMIZATIONS"] = "ON"
        pyctest.BUILD_NAME = "{} OPT".format(pyctest.BUILD_NAME)

    if args.no_mpi:
        build_opts["TIMEMORY_USE_MPI"] = "OFF"
    else:
        pyctest.BUILD_NAME = "{} MPI".format(pyctest.BUILD_NAME)

    if args.no_papi:
        build_opts["TIMEMORY_USE_PAPI"] = "OFF"
    else:
        pyctest.BUILD_NAME = "{} PAPI".format(pyctest.BUILD_NAME)

    if args.arch:
        pyctest.BUILD_NAME = "{} ARCH".format(pyctest.BUILD_NAME)
        build_opts["TIMEMORY_USE_ARCH"] = "ON"

    if args.cuda:
        pyctest.BUILD_NAME = "{} CUDA".format(pyctest.BUILD_NAME)
        build_opts["TIMEMORY_USE_CUDA"] = "ON"
    else:
        build_opts["TIMEMORY_USE_CUDA"] = "OFF"

    if args.profile is not None:
        build_opts["TIMEMORY_USE_GPERF"] = "ON"
        components = "profiler" if args.profile == "cpu" else "tcmalloc"
        build_opts["TIMEMORY_GPERF_COMPONENTS"] = components
        if pyctest.BUILD_TYPE != "RelWithDebInfo":
            warnings.warn(
                "Forcing build type to 'RelWithDebInfo' when gperf is enabled")
            pyctest.BUILD_TYPE = "RelWithDebInfo"
        pyctest.BUILD_NAME = "{} {}".format(
            pyctest.BUILD_NAME, args.profile.upper())

    if args.sanitizer is not None:
        pyctest.BUILD_NAME = "{} {}SAN".format(
            pyctest.BUILD_NAME, args.sanitizer.upper()[0])
        build_opts["SANITIZER_TYPE"] = args.sanitizer
        build_opts["TIMEMORY_USE_SANITIZER"] = "ON"

    if args.static_analysis:
        build_opts["TIMEMORY_USE_CLANG_TIDY"] = "ON"

    if args.coverage:
        gcov_exe = helpers.FindExePath("gcov")
        if gcov_exe is not None:
            pyctest.COVERAGE_COMMAND = "{}".format(gcov_exe)
            build_opts["TIMEMORY_USE_COVERAGE"] = "ON"
            build_opts["TIMEMORY_USE_CALIPER"] = "OFF"
            build_opts["TIMEMORY_BUILD_CALIPER"] = "OFF"
            pyctest.BUILD_NAME = "{} COV".format(pyctest.BUILD_NAME)
            if pyctest.BUILD_TYPE != "Debug":
                warnings.warn(
                    "Forcing build type to 'Debug' when coverage is enabled")
                pyctest.BUILD_TYPE = "Debug"
        pyctest.set("CTEST_CUSTOM_COVERAGE_EXCLUDE", ".*external/.*;/usr/.*")

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
        pyctest.BUILD_COMMAND = "{} -- /MP -A x64".format(
            pyctest.BUILD_COMMAND)

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
    def construct_name(test_name):
        return test_name.replace("_", "-")

    #--------------------------------------------------------------------------#
    # construct a command
    #
    def construct_command(cmd, args):
        global clobber_notes
        _cmd = []
        if args.profile is not None:
            _exe = os.path.basename(cmd[0])
            if args.profile == "cpu":
                _cmd.append(os.path.join(pyctest.BINARY_DIRECTORY,
                                         "gperf-cpu-profile.sh"))
                pyctest.add_note(pyctest.BINARY_DIRECTORY,
                                 "cpu.prof.{}/gperf.0.txt".format(_exe),
                                 clobber=clobber_notes)
                pyctest.add_note(pyctest.BINARY_DIRECTORY,
                                 "cpu.prof.{}/gperf.0.cum.txt".format(_exe),
                                 clobber=False)
                clobber_notes = False
            elif args.profile == "heap":
                _cmd.append(os.path.join(pyctest.BINARY_DIRECTORY,
                                         "gperf-heap-profile.sh"))
                for itr in ["alloc_objects", "alloc_space", "inuse_objects", "inuse_space"]:
                    pyctest.add_note(pyctest.BINARY_DIRECTORY,
                                     "heap.prof.{}/gperf.0.0001.heap.{}.txt".format(
                                         _exe, itr),
                                     clobber=clobber_notes)
                    # make sure all subsequent iterations don't clobber
                    clobber_notes = False
        else:
            _cmd.append("{}/timem".format(pyctest.BINARY_DIRECTORY))
        _cmd.extend(cmd)
        return _cmd

    #--------------------------------------------------------------------------#
    # create tests
    #
    if not args.no_c:
        pyctest.test(construct_name("test-c-timing"),
                     construct_command(["./test_c_timing"], args),
                     {"WORKING_DIRECTORY": pyctest.BINARY_DIRECTORY,
                      "LABELS": pyctest.PROJECT_NAME})

    pyctest.test("test-optional-off", ["./test_optional_off"],
                 {"WORKING_DIRECTORY": pyctest.BINARY_DIRECTORY,
                  "LABELS": pyctest.PROJECT_NAME})

    pyctest.test(construct_name("test-optional-on"),
                 construct_command(["./test_optional_on"], args),
                 {"WORKING_DIRECTORY": pyctest.BINARY_DIRECTORY,
                  "LABELS": pyctest.PROJECT_NAME})

    pyctest.test(construct_name("test-cxx-basic"),
                 construct_command(["./test_cxx_basic"], args),
                 {"WORKING_DIRECTORY": pyctest.BINARY_DIRECTORY,
                  "LABELS": pyctest.PROJECT_NAME})

    pyctest.test(construct_name("test-cxx-tuple"),
                 construct_command(["./test_cxx_tuple"], args),
                 {"WORKING_DIRECTORY": pyctest.BINARY_DIRECTORY,
                  "LABELS": pyctest.PROJECT_NAME,
                  "ENVIRONMENT": "CPUPROFILE_FREQUENCY=2000"})

    pyctest.test(construct_name("test-cxx-overhead"),
                 construct_command(["./test_cxx_overhead"], args),
                 {"WORKING_DIRECTORY": pyctest.BINARY_DIRECTORY,
                  "LABELS": pyctest.PROJECT_NAME})

    pyctest.test(construct_name("test-cpu-roofline"),
                 construct_command(["./test_cpu_roofline"], args),
                 {"WORKING_DIRECTORY": pyctest.BINARY_DIRECTORY,
                  "LABELS": pyctest.PROJECT_NAME,
                  "TIMEOUT": "300"})

    pyctest.test(construct_name("test-caliper"),
                 construct_command(["./test_caliper"], args),
                 {"WORKING_DIRECTORY": pyctest.BINARY_DIRECTORY,
                  "LABELS": pyctest.PROJECT_NAME,
                  "TIMEOUT": "300"})

    pyctest.test(construct_name("test-gotcha"),
                 construct_command(["./test_gotcha"], args),
                 {"WORKING_DIRECTORY": pyctest.BINARY_DIRECTORY,
                  "LABELS": pyctest.PROJECT_NAME,
                  "TIMEOUT": "300"})

    pyctest.test(construct_name("test-c-minimal"),
                 construct_command(["./test_c_minimal"], args),
                 {"WORKING_DIRECTORY": pyctest.BINARY_DIRECTORY,
                  "LABELS": pyctest.PROJECT_NAME,
                  "TIMEOUT": "300"})

    pyctest.test(construct_name("test-cxx-minimal"),
                 construct_command(["./test_cxx_minimal"], args),
                 {"WORKING_DIRECTORY": pyctest.BINARY_DIRECTORY,
                  "LABELS": pyctest.PROJECT_NAME,
                  "TIMEOUT": "300"})

    pyctest.test(construct_name("test-python-minimal"),
                 construct_command(
                     [sys.executable, "./test_python_minimal"], args),
                 {"WORKING_DIRECTORY": pyctest.BINARY_DIRECTORY,
                  "LABELS": pyctest.PROJECT_NAME,
                  "TIMEOUT": "300"})

    pyctest.generate_config(pyctest.BINARY_DIRECTORY)
    pyctest.generate_test_file(os.path.join(pyctest.BINARY_DIRECTORY, "tests"))
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
