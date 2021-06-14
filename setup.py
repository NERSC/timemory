#!/usr/bin/env python

import os
import sys
import argparse
import warnings
import platform
from skbuild import setup
from skbuild.command.install import install as skinstall

# some Cray systems default to static libraries and the build
# will fail because BUILD_SHARED_LIBS will get set to off
if os.environ.get("CRAYPE_VERSION") is not None:
    os.environ["CRAYPE_LINK_TYPE"] = "dynamic"

cmake_args = [
    "-DPYTHON_EXECUTABLE:FILEPATH={}".format(sys.executable),
    "-DPython3_EXECUTABLE:FILEPATH={}".format(sys.executable),
    "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON",
    "-DTIMEMORY_USE_PYTHON:BOOL=ON",
    "-DTIMEMORY_BUILD_PYTHON:BOOL=ON",
]
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-h", "--help", help="Print help", action="store_true")

gotcha_opt = False
if platform.system() == "Linux":
    gotcha_opt = True


def set_cmake_bool_option(opt, enable_opt, disable_opt):
    global cmake_args
    try:
        if enable_opt:
            cmake_args.append("-D{}:BOOL={}".format(opt, "ON"))
        if disable_opt:
            cmake_args.append("-D{}:BOOL={}".format(opt, "OFF"))
    except Exception as e:
        print("Exception: {}".format(e))


def add_arg_bool_option(lc_name, disp_name, default=None):
    global parser
    # enable option
    parser.add_argument(
        "--enable-{}".format(lc_name),
        action="store_true",
        default=default,
        help="Explicitly enable {} build".format(disp_name),
    )
    # disable option
    parser.add_argument(
        "--disable-{}".format(lc_name),
        action="store_true",
        help="Explicitly disable {} build".format(disp_name),
    )


# add options
add_arg_bool_option("c", "TIMEMORY_BUILD_C")
add_arg_bool_option("tools", "TIMEMORY_BUILD_TOOLS", default=True)
add_arg_bool_option("mpi", "TIMEMORY_USE_MPI")
add_arg_bool_option("nccl", "TIMEMORY_USE_NCCL")
add_arg_bool_option("upcxx", "TIMEMORY_USE_UPCXX")
add_arg_bool_option("cuda", "TIMEMORY_USE_CUDA")
add_arg_bool_option("cupti", "TIMEMORY_USE_CUPTI")
add_arg_bool_option("papi", "TIMEMORY_USE_PAPI")
add_arg_bool_option("arch", "TIMEMORY_USE_ARCH")
add_arg_bool_option("ompt", "TIMEMORY_USE_OMPT")
add_arg_bool_option("gotcha", "TIMEMORY_USE_GOTCHA", default=gotcha_opt)
add_arg_bool_option("kokkos", "TIMEMORY_BUILD_KOKKOS_TOOLS", default=False)
add_arg_bool_option("dyninst", "TIMEMORY_BUILD_DYNINST_TOOLS")
add_arg_bool_option("tau", "TIMEMORY_USE_TAU")
add_arg_bool_option("caliper", "TIMEMORY_USE_CALIPER")
add_arg_bool_option("likwid", "TIMEMORY_USE_LIKWID")
add_arg_bool_option("gperftools", "TIMEMORY_USE_GPERFTOOLS")
add_arg_bool_option("vtune", "TIMEMORY_USE_VTUNE")
add_arg_bool_option("build-caliper", "TIMEMORY_BUILD_CALIPER")
add_arg_bool_option("build-gotcha", "TIMEMORY_BUILD_GOTCHA")
add_arg_bool_option("pybind-install", "PYBIND11_INSTALL")
add_arg_bool_option("build-testing", "TIMEMORY_BUILD_TESTING")
parser.add_argument(
    "--cxx-standard",
    default=14,
    type=int,
    choices=[14, 17, 20],
    help="Set C++ language standard",
)
parser.add_argument(
    "--cmake-args",
    default=[],
    type=str,
    nargs="*",
    help="{}{}".format(
        "Pass arguments to cmake. Use w/ pip installations and --install-option, e.g. ",
        '--install-option=--cmake-args="-DTIMEMORY_BUILD_LTO=ON -DCMAKE_UNITY_BUILD=OFF"',
    ),
)

args, left = parser.parse_known_args()
# if help was requested, print these options and then add '--help' back
# into arguments so that the skbuild/setuptools argparse catches it
if args.help:
    parser.print_help()
    left.append("--help")
sys.argv = sys.argv[:1] + left

runtime_req_file = (
    ".requirements/mpi_runtime.txt"
    if args.enable_mpi and not args.disable_mpi
    else ".requirements/runtime.txt"
)

set_cmake_bool_option("TIMEMORY_BUILD_C", args.enable_c, args.disable_c)
set_cmake_bool_option(
    "TIMEMORY_BUILD_TOOLS", args.enable_tools, args.disable_tools
)
set_cmake_bool_option("TIMEMORY_USE_MPI", args.enable_mpi, args.disable_mpi)
set_cmake_bool_option("TIMEMORY_USE_NCCL", args.enable_nccl, args.disable_nccl)
set_cmake_bool_option(
    "TIMEMORY_USE_UPCXX", args.enable_upcxx, args.disable_upcxx
)
set_cmake_bool_option(
    "TIMEMORY_USE_GOTCHA", args.enable_gotcha, args.disable_gotcha
)
set_cmake_bool_option("TIMEMORY_USE_CUDA", args.enable_cuda, args.disable_cuda)
set_cmake_bool_option(
    "TIMEMORY_USE_CUPTI", args.enable_cupti, args.disable_cupti
)
set_cmake_bool_option("TIMEMORY_USE_PAPI", args.enable_papi, args.disable_papi)
set_cmake_bool_option("TIMEMORY_USE_ARCH", args.enable_arch, args.disable_arch)
set_cmake_bool_option("TIMEMORY_USE_OMPT", args.enable_ompt, args.disable_ompt)
set_cmake_bool_option(
    "TIMEMORY_USE_DYNINST", args.enable_dyninst, args.disable_dyninst
)
set_cmake_bool_option(
    "TIMEMORY_BUILD_OMPT_LIBRARY",
    args.enable_ompt and args.enable_tools,
    args.disable_ompt or args.disable_tools,
)
set_cmake_bool_option(
    "TIMEMORY_BUILD_MPIP_LIBRARY",
    args.enable_gotcha and args.enable_mpi and args.enable_tools,
    args.disable_gotcha or args.disable_mpi or args.disable_tools,
)
set_cmake_bool_option(
    "TIMEMORY_BUILD_NCCLP_LIBRARY",
    args.enable_gotcha and args.enable_nccl and args.enable_tools,
    args.disable_gotcha or args.disable_nccl or args.disable_tools,
)
set_cmake_bool_option(
    "TIMEMORY_BUILD_DYNINST_TOOLS", args.enable_dyninst, args.disable_dyninst
)
set_cmake_bool_option(
    "TIMEMORY_BUILD_KOKKOS_TOOLS", args.enable_kokkos, args.disable_kokkos
)
set_cmake_bool_option("TIMEMORY_USE_TAU", args.enable_tau, args.disable_tau)
set_cmake_bool_option(
    "TIMEMORY_USE_CALIPER", args.enable_caliper, args.disable_caliper
)
set_cmake_bool_option(
    "TIMEMORY_USE_LIKWID", args.enable_likwid, args.disable_likwid
)
set_cmake_bool_option(
    "TIMEMORY_USE_VTUNE", args.enable_vtune, args.disable_vtune
)
set_cmake_bool_option(
    "TIMEMORY_USE_GPERFTOOLS", args.enable_gperftools, args.disable_gperftools
)
set_cmake_bool_option(
    "TIMEMORY_BUILD_CALIPER",
    args.enable_build_caliper,
    args.disable_build_caliper,
)
set_cmake_bool_option(
    "TIMEMORY_BUILD_GOTCHA", args.enable_build_gotcha, args.disable_build_gotcha
)
set_cmake_bool_option(
    "PYBIND11_INSTALL", args.enable_pybind_install, args.disable_pybind_install
)
set_cmake_bool_option(
    "TIMEMORY_BUILD_TESTING",
    args.enable_build_testing,
    args.disable_build_testing,
)
cmake_args.append("-DCMAKE_CXX_STANDARD={}".format(args.cxx_standard))

for itr in args.cmake_args:
    cmake_args += itr.split()

gotcha_opt = False
if platform.system() == "Darwin":
    # scikit-build will set this to 10.6 and C++ compiler check will fail
    version = platform.mac_ver()[0].split(".")
    version = ".".join([version[0], version[1]])
    cmake_args += ["-DCMAKE_OSX_DEPLOYMENT_TARGET={}".format(version)]
elif platform.system() == "Linux":
    gotcha_opt = True


# --------------------------------------------------------------------------- #
#
def get_project_version():
    # open "VERSION"
    with open(os.path.join(os.getcwd(), "VERSION"), "r") as f:
        data = f.read().replace("\n", "")
    # make sure is string
    if isinstance(data, list) or isinstance(data, tuple):
        return data[0]
    else:
        return data


# --------------------------------------------------------------------------- #
#
def get_long_description():
    long_descript = ""
    try:
        long_descript = open("README.md").read()
    except Exception:
        long_descript = ""
    return long_descript


# --------------------------------------------------------------------------- #
class custom_install(skinstall):
    """
    Custom installation
    """

    def __init__(self, *args, **kwargs):
        skinstall.__init__(self, *args, **kwargs)

    def run(self):
        print("\n\n\tRunning install...")
        print("\t\t {:10} : {}".format("base", self.install_base))
        print("\t\t {:10} : {}".format("purelib", self.install_purelib))
        print("\t\t {:10} : {}".format("platlib", self.install_platlib))
        print("\t\t {:10} : {}".format("headers", self.install_headers))
        print("\t\t {:10} : {}".format("lib", self.install_lib))
        print("\t\t {:10} : {}".format("scripts", self.install_scripts))
        print("\t\t {:10} : {}".format("data", self.install_data))
        print("\t\t {:10} : {}".format("userbase", self.install_userbase))
        print("\t\t {:10} : {}".format("usersite", self.install_usersite))
        print("\n\n")
        skinstall.run(self)
        for itr in self.get_outputs():
            print('installed file : "{}"'.format(itr))


# --------------------------------------------------------------------------- #
#
def parse_requirements(fname="requirements.txt"):
    _req = []
    requirements = []
    # read in the initial set of requirements
    with open(fname, "r") as fp:
        _req = list(filter(bool, (line.strip() for line in fp)))
    # look for entries which read other files
    for itr in _req:
        if itr.startswith("-r "):
            # read another file
            for fitr in itr.split(" "):
                if os.path.exists(fitr):
                    requirements.extend(parse_requirements(fitr))
        else:
            # append package
            requirements.append(itr)
    # return the requirements
    return requirements


# suppress:
#  "setuptools_scm/git.py:68: UserWarning: "/.../<PACKAGE>"
#       is shallow and may cause errors"
# since 'error' in output causes CDash to interpret warning as error
with warnings.catch_warnings():
    print("CMake arguments: {}".format(" ".join(cmake_args)))
    setup(
        name="timemory",
        packages=["timemory"],
        version=get_project_version(),
        cmake_args=cmake_args,
        cmake_languages=("C", "CXX"),
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        install_requires=parse_requirements(runtime_req_file),
        extras_require={
            "all": parse_requirements("requirements.txt")
            + parse_requirements(".requirements/mpi_runtime.txt"),
            "mpi": parse_requirements(".requirements/mpi_runtime.txt"),
            "build": parse_requirements(".requirements/build.txt"),
        },
        python_requires=">=3.6",
        cmdclass=dict(install=custom_install),
        entry_points={
            "console_scripts": [
                "timemory-plotter=timemory.plotting.__main__:try_plot",
                "timemory-roofline=timemory.roofline.__main__:try_plot",
                "timemory-analyze=timemory.analyze.__main__:try_analyze",
                "timemory-python-line-profiler=timemory.line_profiler.__main__:main",
                "timemory-python-profiler=timemory.profiler.__main__:main",
                "timemory-python-trace=timemory.trace.__main__:main",
            ],
        },
    )
