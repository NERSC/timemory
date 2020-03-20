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
import pyctest.pyctest as pyct
import pyctest.pycmake as pycm
import pyctest.helpers as helpers

clobber_notes = True


#------------------------------------------------------------------------------#
def get_branch(wd=pyct.SOURCE_DIRECTORY):
    cmd = pyct.command(["git", "show", "-s", "--pretty=%d", "HEAD"])
    cmd.SetOutputStripTrailingWhitespace(True)
    cmd.SetWorkingDirectory(wd)
    cmd.Execute()
    branch = cmd.Output()
    branch = branch.split(" ")
    if branch:
        branch = branch[len(branch)-1]
        branch = branch.strip(")")
    if not branch:
        branch = pyct.GetGitBranch(wd)
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
    parser.add_argument("--tools", help="TIMEMORY_BUILD_TOOLS=ON",
                        default=False, action='store_true')
    parser.add_argument("--tau", help="TIMEMORY_USE_TAU=ON",
                        default=False, action='store_true')
    parser.add_argument("--mpip", help="TIMEMORY_BUILD_MPIP=ON",
                        default=False, action='store_true')
    parser.add_argument("--cuda", help="TIMEMORY_USE_CUDA=ON",
                        default=False, action='store_true')
    parser.add_argument("--nvtx", help="TIMEMORY_USE_NVTX=ON",
                        default=False, action='store_true')
    parser.add_argument("--cupti", help="TIMEMORY_USE_CUPTI=ON",
                        default=False, action='store_true')
    parser.add_argument("--upcxx", help="TIMEMORY_USE_UPCXX=ON",
                        default=False, action='store_true')
    parser.add_argument("--gotcha", help="TIMEMORY_USE_GOTCHA=ON",
                        default=False, action='store_true')
    parser.add_argument("--gperftools", help="TIMEMORY_USE_GPERFTOOLS=ON",
                        default=False, action='store_true')
    parser.add_argument("--caliper", help="TIMEMORY_USE_CALIPER=ON",
                        default=False, action='store_true')
    parser.add_argument("--likwid", help="TIMEMORY_USE_LIKWID=ON",
                        default=False, action='store_true')
    parser.add_argument("--papi", help="TIMEMORY_USE_PAPI=ON",
                        default=False, action='store_true')
    parser.add_argument("--mpi", help="TIMEMORY_USE_MPI=ON",
                        default=False, action='store_true')
    parser.add_argument("--mpi-init", help="TIMEMORY_USE_MPI_INIT=ON",
                        default=False, action='store_true')
    parser.add_argument("--python", help="TIMEMORY_BUILD_PYTHON=ON",
                        default=False, action='store_true')
    parser.add_argument("--ompt", help="TIMEMORY_BUILD_OMPT=ON",
                        default=False, action='store_true')
    parser.add_argument("--kokkos", help="TIMEMORY_BUILD_KOKKOS_TOOLS=ON",
                        default=False, action='store_true')
    parser.add_argument("--dyninst", help="TIMEMORY_USE_DYNINST=ON",
                        default=False, action='store_true')
    parser.add_argument("--extra-optimizations",
                        help="TIMEMORY_BUILD_EXTRA_OPTIMIZATIONS=ON",
                        default=False, action='store_true')
    parser.add_argument("--lto", help="TIMEMORY_BUILD_LTO=ON",
                        default=False, action='store_true')
    parser.add_argument("--developer", help="TIMEMORY_BUILD_DEVELOPER=ON",
                        default=False, action='store_true')
    parser.add_argument("--xray", help="TIMEMORY_BUILD_XRAY=ON",
                        default=False, action='store_true')
    parser.add_argument("--stats", help="TIMEMORY_USE_STATISTICS=ON",
                        default=False, action='store_true')
    parser.add_argument("--timing", help="TIMEMORY_USE_COMPILE_TIMING=ON",
                        default=False, action='store_true')
    parser.add_argument("--build-libs", help="Build library type(s)", default=("shared"),
                        nargs='*', type=str, choices=("static", "shared"))
    parser.add_argument("--tls-model", help="Thread-local static model",
                        default=('global-dynamic'), type=str,
                        choices=('global-dynamic', 'local-dynamic', 'initial-exec', 'local-exec'))
    parser.add_argument("--cxx-standard", help="C++ standard", type=str,
                        default="17", choices=("14", "17", "20"))
    parser.add_argument("--generate", help="Generate the tests only",
                        action='store_true')
    parser.add_argument("-j", "--cpu-count", type=int, default=mp.cpu_count(),
                        help="Parallel build jobs to run")

    args = parser.parse_args()

    if os.environ.get("CTEST_SITE") is not None:
        pyct.set("CTEST_SITE", "{}".format(os.environ.get("CTEST_SITE")))

    if os.path.exists(os.path.join(pyct.BINARY_DIRECTORY, "CMakeCache.txt")):
        from pyctest import cmake_executable as cm
        from pyctest import version_info as _pyctest_version
        if (_pyctest_version[0] == 0 and
                _pyctest_version[1] == 0 and
                _pyctest_version[2] < 11):
            cmd = pyct.command(
                [cm, "--build", pyct.BINARY_DIRECTORY, "--target", "clean"])
            cmd.SetWorkingDirectory(pyct.BINARY_DIRECTORY)
            cmd.SetOutputQuiet(True)
            cmd.SetErrorQuiet(True)
            cmd.Execute()
        else:
            from pyctest.cmake import CMake
            CMake("--build", pyct.BINARY_DIRECTORY, "--target", "clean")
        helpers.RemovePath(os.path.join(
            pyct.BINARY_DIRECTORY, "CMakeCache.txt"))

    if platform.system() != "Linux":
        args.papi = False

    os.environ["PYCTEST_TESTING"] = "ON"
    os.environ["TIMEMORY_PLOT_OUTPUT"] = "ON"

    # update PYTHONPATH for the unit tests
    pypath = os.environ.get("PYTHONPATH", "").split(":")
    pypath = [pyct.BINARY_DIRECTORY] + pypath
    os.environ["PYTHONPATH"] = ":".join(pypath)

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
    cmd = pyct.command([os.environ["CXX"], "-dumpversion"])
    cmd.SetOutputStripTrailingWhitespace(True)
    cmd.Execute()
    compiler_version = cmd.Output()

    #--------------------------------------------------------------------------#
    # Set the build name
    #
    pyct.BUILD_NAME = "{} {} {} {} {} {}".format(
        get_branch(pyct.SOURCE_DIRECTORY),
        platform.uname()[0],
        helpers.GetSystemVersionInfo(),
        platform.uname()[4],
        os.path.basename(os.path.realpath(os.environ["CXX"])),
        compiler_version)
    pyct.BUILD_NAME = '-'.join(pyct.BUILD_NAME.split())

    #--------------------------------------------------------------------------#
    #   build specifications
    #
    build_opts = {
        "BUILD_SHARED_LIBS": "ON" if "shared" in args.build_libs else "OFF",
        "BUILD_STATIC_LIBS": "ON" if "static" in args.build_libs else "OFF",
        "CMAKE_CXX_STANDARD": "{}".format(args.cxx_standard),
        "TIMEMORY_TLS_MODEL": "{}".format(args.tls_model),
        "TIMEMORY_CCACHE_BUILD": "OFF",
        "TIMEMORY_BUILD_LTO": "ON" if args.lto else "OFF",
        "TIMEMORY_BUILD_OMPT": "ON" if args.ompt else "OFF",
        "TIMEMORY_BUILD_TOOLS": "ON" if args.tools else "OFF",
        "TIMEMORY_BUILD_GOTCHA": "ON" if args.gotcha else "OFF",
        "TIMEMORY_BUILD_PYTHON": "ON" if args.python else "OFF",
        "TIMEMORY_BUILD_CALIPER": "ON" if args.caliper else "OFF",
        "TIMEMORY_BUILD_DEVELOPER": "ON" if args.developer else "OFF",
        "TIMEMORY_BUILD_TESTING": "ON",
        "TIMEMORY_BUILD_EXTRA_OPTIMIZATIONS": "ON" if args.extra_optimizations else "OFF",
        "TIMEMORY_USE_MPI": "ON" if args.mpi else "OFF",
        "TIMEMORY_USE_TAU": "ON" if args.tau else "OFF",
        "TIMEMORY_USE_ARCH": "ON" if args.arch else "OFF",
        "TIMEMORY_USE_PAPI": "ON" if args.papi else "OFF",
        "TIMEMORY_USE_CUDA": "ON" if args.cuda else "OFF",
        "TIMEMORY_USE_NVTX": "ON" if args.nvtx else "OFF",
        "TIMEMORY_USE_OMPT": "ON" if args.ompt else "OFF",
        "TIMEMORY_USE_XRAY": "ON" if args.xray else "OFF",
        "TIMEMORY_USE_CUPTI": "ON" if args.cupti else "OFF",
        "TIMEMORY_USE_UPCXX": "ON" if args.upcxx else "OFF",
        "TIMEMORY_USE_LIKWID": "ON" if args.likwid else "OFF",
        "TIMEMORY_USE_GOTCHA": "ON" if args.gotcha else "OFF",
        "TIMEMORY_USE_PYTHON": "ON" if args.python else "OFF",
        "TIMEMORY_USE_CALIPER": "ON" if args.caliper else "OFF",
        "TIMEMORY_USE_COVERAGE": "ON" if args.coverage else "OFF",
        "TIMEMORY_USE_GPERFTOOLS": "ON" if args.gperftools else "OFF",
        "TIMEMORY_USE_STATISTICS": "ON" if args.stats else "OFF",
        "TIMEMORY_USE_COMPILE_TIMING": "ON" if args.timing else "OFF",
        "TIMEMORY_USE_MPI_INIT": "ON" if args.mpi_init else "OFF",
        "TIMEMORY_USE_SANITIZER": "OFF",
        "TIMEMORY_USE_CLANG_TIDY": "ON" if args.static_analysis else "OFF",
        "USE_PAPI": "ON" if args.papi else "OFF",
        "USE_MPI": "ON" if args.mpi else "OFF",
        "USE_CALIPER": "ON" if args.caliper else "OFF",
    }

    if args.ompt:
        build_opts["OPENMP_ENABLE_LIBOMPTARGET"] = "OFF"

    if args.tools:
        build_opts["TIMEMORY_BUILD_MPIP"] = "ON" if (
            args.mpi and args.mpip) else "OFF"
        build_opts["TIMEMORY_BUILD_KOKKOS_TOOLS"] = "ON" if args.kokkos else "OFF"
        build_opts["TIMEMORY_BUILD_DYNINST_TOOLS"] = "ON" if args.dyninst else "OFF"

    if args.python:
        pyver = "{}.{}.{}".format(
            sys.version_info[0], sys.version_info[1], sys.version_info[2])
        pyct.BUILD_NAME = "{} PY-{}".format(pyct.BUILD_NAME, pyver)

    if args.profile is not None:
        build_opts["TIMEMORY_USE_GPERFTOOLS"] = "ON"
        components = "profiler" if args.profile == "cpu" else "tcmalloc"
        build_opts["TIMEMORY_gperftools_COMPONENTS"] = components
        pyct.BUILD_NAME = "{} {}".format(
            pyct.BUILD_NAME, args.profile.upper())

    if args.sanitizer is not None:
        pyct.BUILD_NAME = "{} {}SAN".format(
            pyct.BUILD_NAME, args.sanitizer.upper()[0])
        build_opts["SANITIZER_TYPE"] = args.sanitizer
        build_opts["TIMEMORY_USE_SANITIZER"] = "ON"

    if args.coverage:
        gcov_exe = helpers.FindExePath("gcov")
        if gcov_exe is not None:
            pyct.COVERAGE_COMMAND = "{}".format(gcov_exe)
            build_opts["TIMEMORY_USE_COVERAGE"] = "ON"
            pyct.BUILD_NAME = "{} COV".format(pyct.BUILD_NAME)
            if pyct.BUILD_TYPE != "Debug":
                warnings.warn(
                    "Forcing build type to 'Debug' when coverage is enabled")
                pyct.BUILD_TYPE = "Debug"
        else:
            build_opts["TIMEMORY_USE_COVERAGE"] = "OFF"
        pyct.set("CTEST_CUSTOM_COVERAGE_EXCLUDE", ".*external/.*;/usr/.*")

    # Use the options to create a build name with configuration
    build_name = set()
    mangled_tags = {"EXTRA_OPTIMIZATIONS": "OPT", "KOKKOS_TOOLS": "KOKKOS",
                    "DYNINST_TOOLS": "DYNINST"}
    exclude_keys = ("TESTING", "EXAMPLES", "GOOGLE_TEST", "CCACHE_BUILD",
                    "gperftools_COMPONENTS")
    for opt_key, opt_val in build_opts.items():
        tag = None
        key = None
        if opt_val == "OFF" or opt_val is None:
            continue
        else:
            if "TIMEMORY_BUILD_" in opt_key:
                tag = opt_key.replace("TIMEMORY_BUILD_", "")
                key = tag
            elif "TIMEMORY_USE_" in opt_key:
                tag = opt_key.replace("TIMEMORY_USE_", "")
                key = tag
            elif "TIMEMORY_" in opt_key:
                key = opt_key.replace("TIMEMORY_", "")
                tag = "{}_{}".format(key, opt_val)

        # if valid and turned on
        if tag is not None and key is not None and key not in exclude_keys:
            tag = mangled_tags.get(tag, tag)
            build_name.add(tag)

    build_name = sorted(build_name)
    pyct.BUILD_NAME += " {}".format(" ".join(build_name))

    # split and join with dashes
    pyct.BUILD_NAME = '-'.join(pyct.BUILD_NAME.replace('/', '-').split())

    # default options
    cmake_args = "-DCMAKE_BUILD_TYPE={} -DTIMEMORY_BUILD_EXAMPLES=ON".format(
        pyct.BUILD_TYPE)

    # customized from args
    for key, val in build_opts.items():
        cmake_args = "{} -D{}={}".format(cmake_args, key, val)

    #--------------------------------------------------------------------------#
    # how to build the code
    #
    ctest_cmake_cmd = "${CTEST_CMAKE_COMMAND}"
    pyct.CONFIGURE_COMMAND = "{} {} {}".format(
        ctest_cmake_cmd, cmake_args, pyct.SOURCE_DIRECTORY)

    #--------------------------------------------------------------------------#
    # how to build the code
    #
    pyct.BUILD_COMMAND = "{} --build {} --target all".format(
        ctest_cmake_cmd, pyct.BINARY_DIRECTORY)

    #--------------------------------------------------------------------------#
    # parallel build
    #
    if platform.system() != "Windows":
        pyct.BUILD_COMMAND = "{} -- -j{} {}".format(
            pyct.BUILD_COMMAND, args.cpu_count, " ".join(pycm.ARGUMENTS))
    else:
        pyct.BUILD_COMMAND = "{} -- /MP -A x64".format(
            pyct.BUILD_COMMAND)

    #--------------------------------------------------------------------------#
    # how to update the code
    #
    git_exe = helpers.FindExePath("git")
    pyct.UPDATE_COMMAND = "{}".format(git_exe)
    pyct.set("CTEST_UPDATE_TYPE", "git")
    pyct.set("CTEST_GIT_COMMAND", "{}".format(git_exe))

    #--------------------------------------------------------------------------#
    # find the CTEST_TOKEN_FILE
    #
    if args.pyctest_token_file is None and args.pyctest_token is None:
        home = helpers.GetHomePath()
        if home is not None:
            token_path = os.path.join(
                home, os.path.join(".tokens", "nersc-cdash"))
            if os.path.exists(token_path):
                pyct.set("CTEST_TOKEN_FILE", token_path)

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
                _cmd.append(os.path.join(pyct.BINARY_DIRECTORY,
                                         "gperf-cpu-profile.sh"))
                pyct.add_note(pyct.BINARY_DIRECTORY,
                              "cpu.prof.{}/gperf.0.txt".format(_exe),
                              clobber=clobber_notes)
                pyct.add_note(pyct.BINARY_DIRECTORY,
                              "cpu.prof.{}/gperf.0.cum.txt".format(_exe),
                              clobber=False)
                clobber_notes = False
            elif args.profile == "heap":
                _cmd.append(os.path.join(pyct.BINARY_DIRECTORY,
                                         "gperf-heap-profile.sh"))
                for itr in ["alloc_objects", "alloc_space", "inuse_objects", "inuse_space"]:
                    pyct.add_note(pyct.BINARY_DIRECTORY,
                                  "heap.prof.{}/gperf.0.0001.heap.{}.txt".format(
                                      _exe, itr),
                                  clobber=clobber_notes)
                    # make sure all subsequent iterations don't clobber
                    clobber_notes = False
        _cmd.extend(cmd)
        return _cmd

    #--------------------------------------------------------------------------#
    # construct a command
    #
    def construct_roofline_command(cmd, dir, extra_opts=[]):
        _cmd = [sys.executable, '-m', 'timemory.roofline', '-e',
                '-D', dir, '--format', 'png']
        _cmd.extend(extra_opts)
        _cmd.extend(['--'])
        _cmd.extend(cmd)
        return _cmd

    #--------------------------------------------------------------------------#
    # create tests
    #
    test_env = ";".join(["CPUPROFILE_FREQUENCY=200",
                         "CPUPROFILE_REALTIME=1",
                         "CALI_CONFIG_PROFILE=runtime-report",
                         "TIMEMORY_DART_OUTPUT=ON",
                         "TIMEMORY_DART_COUNT=1",
                         "TIMEMORY_PLOT_OUTPUT=ON"])

    pyct.test(construct_name("ex-assemble"),
              construct_command(["./ex_assemble"], args),
              {"WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
               "LABELS": pyct.PROJECT_NAME,
               "TIMEOUT": "300",
               "ENVIRONMENT": test_env})

    pyct.test(construct_name("ex-optional-off"),
              construct_command(["./ex_optional_off"], args),
              {"WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
               "LABELS": pyct.PROJECT_NAME,
               "TIMEOUT": "300",
               "ENVIRONMENT": test_env})

    pyct.test(construct_name("ex-cxx-overhead"),
              construct_command(["./ex_cxx_overhead"], args),
              {"WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
               "LABELS": pyct.PROJECT_NAME,
               "TIMEOUT": "600",
               "ENVIRONMENT": test_env})

    if args.cuda:
        pyct.test(construct_name("ex-cuda-event"),
                  ["./ex_cuda_event"],
                  {"WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                   "LABELS": pyct.PROJECT_NAME,
                   "TIMEOUT": "300",
                   "ENVIRONMENT": test_env})

    pyct.test(construct_name("ex-cxx-minimal"),
              construct_command(["./ex_cxx_minimal"], args),
              {"WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
               "LABELS": pyct.PROJECT_NAME,
               "TIMEOUT": "300",
               "ENVIRONMENT": test_env})

    pyct.test(construct_name("ex-c-minimal-library-overload"),
              construct_command(["./ex_c_minimal_library_overload"], args),
              {"WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
               "LABELS": pyct.PROJECT_NAME,
               "TIMEOUT": "300",
               "ENVIRONMENT": test_env})

    pyct.test(construct_name("ex-c-timing"),
              construct_command(["./ex_c_timing"], args),
              {"WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
               "LABELS": pyct.PROJECT_NAME,
               "TIMEOUT": "300",
               "ENVIRONMENT": test_env})

    pyct.test(construct_name("ex-cxx-minimal-library"),
              construct_command(["./ex_cxx_minimal_library"], args),
              {"WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
               "LABELS": pyct.PROJECT_NAME,
               "TIMEOUT": "300",
               "ENVIRONMENT": test_env})

    pyct.test(construct_name("ex-optional-on"),
              construct_command(["./ex_optional_on"], args),
              {"WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
               "LABELS": pyct.PROJECT_NAME,
               "TIMEOUT": "300",
               "ENVIRONMENT": test_env})

    pyct.test(construct_name("ex-c-minimal-library"),
              construct_command(["./ex_c_minimal_library"], args),
              {"WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
               "LABELS": pyct.PROJECT_NAME,
               "TIMEOUT": "300",
               "ENVIRONMENT": test_env})

    pyct.test(construct_name("ex-ert"),
              construct_command(["./ex_ert"], args),
              {"WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
               "LABELS": pyct.PROJECT_NAME,
               "TIMEOUT": "600",
               "ENVIRONMENT": test_env})

    pyct.test(construct_name("ex-cxx-tuple"),
              construct_command(["./ex_cxx_tuple"], args),
              {"WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
               "LABELS": pyct.PROJECT_NAME,
               "TIMEOUT": "300",
               "ENVIRONMENT": test_env})

    if args.gotcha:
        pyct.test(construct_name("ex-gotcha"),
                  construct_command(["./ex_gotcha"], args),
                  {"WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                   "LABELS": pyct.PROJECT_NAME,
                   "TIMEOUT": "300",
                   "ENVIRONMENT": test_env})

        pyct.test(construct_name("ex-gotcha-replacement"),
                  construct_command(["./ex_gotcha_replacement"], args),
                  {"WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                   "LABELS": pyct.PROJECT_NAME,
                   "TIMEOUT": "300",
                   "ENVIRONMENT": test_env})

        if args.mpi:
            pyct.test(construct_name("ex-gotcha-mpi"),
                      construct_command(["./ex_gotcha_mpi"], args),
                      {"WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                       "LABELS": pyct.PROJECT_NAME,
                       "TIMEOUT": "300",
                       "ENVIRONMENT": test_env})

    if args.python:
        pyct.test(construct_name("ex-python-caliper"),
                  construct_command(["./ex_python_caliper"], args),
                  {"WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                   "LABELS": pyct.PROJECT_NAME,
                   "TIMEOUT": "300",
                   "ENVIRONMENT": test_env})

    pyct.test(construct_name("ex-caliper"),
              construct_command(["./ex_caliper"], args),
              {"WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
               "LABELS": pyct.PROJECT_NAME,
               "TIMEOUT": "300",
               "ENVIRONMENT": test_env})

    pyct.test(construct_name("ex-c-minimal"),
              construct_command(["./ex_c_minimal"], args),
              {"WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
               "LABELS": pyct.PROJECT_NAME,
               "TIMEOUT": "300",
               "ENVIRONMENT": test_env})

    pyct.test(construct_name("ex-cxx-minimal-library-overload"),
              construct_command(["./ex_cxx_minimal_library_overload"], args),
              {"WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
               "LABELS": pyct.PROJECT_NAME,
               "TIMEOUT": "300",
               "ENVIRONMENT": test_env})

    pyct.test(construct_name("ex-cxx-basic"),
              construct_command(["./ex_cxx_basic"], args),
              {"WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
               "LABELS": pyct.PROJECT_NAME,
               "TIMEOUT": "300",
               "ENVIRONMENT": test_env})

    pyct.test(construct_name("ex-statistics"),
              construct_command(["./ex_cxx_statistics"], args),
              {"WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
               "LABELS": pyct.PROJECT_NAME,
               "TIMEOUT": "300",
               "ENVIRONMENT": test_env})

    if args.python:
        pyct.test(construct_name("ex-python-minimal"),
                  construct_command(["./ex_python_minimal"], args),
                  {"WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                   "LABELS": pyct.PROJECT_NAME,
                   "TIMEOUT": "480",
                   "ENVIRONMENT": test_env})

    if args.likwid:
        pyct.test(construct_name("ex-likwid"),
                  construct_command(["./ex_likwid"], args),
                  {"WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                   "LABELS": pyct.PROJECT_NAME,
                   "TIMEOUT": "300",
                   "ENVIRONMENT": test_env})

        if args.python:
            pyct.test(construct_name("ex-python-likwid"),
                      construct_command(["./ex_python_likwid"], args),
                      {"WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                       "LABELS": pyct.PROJECT_NAME,
                       "TIMEOUT": "300",
                       "ENVIRONMENT": test_env})

    if not args.python:
        pyct.test(construct_name("ex-cpu-roofline"),
                  construct_roofline_command(["./ex_cpu_roofline"], 'cpu-roofline',
                                             ['-t', 'cpu_roofline']),
                  {"WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                   "LABELS": pyct.PROJECT_NAME,
                   "TIMEOUT": "900",
                   "ENVIRONMENT": test_env})

        pyct.test(construct_name("ex-cpu-roofline.sp"),
                  construct_roofline_command(["./ex_cpu_roofline.sp"], 'cpu-roofline.sp',
                                             ['-t', 'cpu_roofline']),
                  {"WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                   "LABELS": pyct.PROJECT_NAME,
                   "TIMEOUT": "900",
                   "ENVIRONMENT": test_env})

        if args.cupti:
            pyct.test(construct_name("ex-gpu-roofline"),
                      construct_roofline_command(["./ex_gpu_roofline"], 'gpu-roofline',
                                                 ['-t', 'gpu_roofline']),
                      {"WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                       "LABELS": pyct.PROJECT_NAME,
                       "TIMEOUT": "900",
                       "ENVIRONMENT": test_env})

    pyct.generate_config(pyct.BINARY_DIRECTORY)
    pyct.generate_test_file(os.path.join(pyct.BINARY_DIRECTORY, "tests"))
    if not args.generate:
        pyct.run(pyct.ARGUMENTS, pyct.BINARY_DIRECTORY)
        if args.coverage:
            script = os.path.join(pyct.SOURCE_DIRECTORY, "cmake",
                                  "Scripts", "submit-coverage.sh")
            cov = pyct.command([script, pyct.BINARY_DIRECTORY])
            cov.SetWorkingDirectory(pyct.SOURCE_DIRECTORY)
            cov.Execute()
            print("{}".format(cov.Output()))
    else:
        print("BUILD_NAME: {}".format(pyct.BUILD_NAME))


#------------------------------------------------------------------------------#
#
if __name__ == "__main__":

    try:

        run_pyctest()

    except Exception as e:
        print('Error running pyctest - {}'.format(e))
        exc_type, exc_value, exc_trback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_trback, limit=10)
        sys.exit(1)

    sys.exit(0)
