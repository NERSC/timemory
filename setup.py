#!/usr/bin/env python

import os
import sys
import glob
import shutil
import argparse
import warnings
import platform
import subprocess as sp
from setuptools import Command
from setuptools import find_packages
from skbuild import setup
from skbuild.setuptools_wrap import create_skbuild_argparser
from skbuild.command.install import install as skinstall
from distutils.command.install import install


cmake_args = ['-DPYTHON_EXECUTABLE={}'.format(sys.executable),
              '-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON']
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-h", "--help", help="Print help", action='store_true')


def set_cmake_bool_option(opt, enable_opt, disable_opt):
    global cmake_args
    try:
        if enable_opt:
            cmake_args.append("-D{}:BOOL={}".format(opt, "ON"))
        if disable_opt:
            cmake_args.append("-D{}:BOOL={}".format(opt, "OFF"))
    except Exception as e:
        print("Exception: {}".format(e))

def add_arg_bool_option(lc_name, disp_name):
    global parser
    # enable option
    parser.add_argument("--enable-{}".format(lc_name), action='store_true',
                        help="Explicitly enable {} build".format(disp_name))
    # disable option
    parser.add_argument("--disable-{}".format(lc_name), action='store_true',
                        help="Explicitly disable {} build".format(disp_name))


add_arg_bool_option("mpi", "TIMEMORY_USE_MPI")
add_arg_bool_option("tau", "TIMEMORY_USE_TAU")
add_arg_bool_option("caliper", "TIMEMORY_USE_CALIPER")
add_arg_bool_option("upcxx", "TIMEMORY_USE_UPCXX")
add_arg_bool_option("cuda", "TIMEMORY_USE_CUDA")
add_arg_bool_option("cupti", "TIMEMORY_USE_CUPTI")
add_arg_bool_option("papi", "TIMEMORY_USE_PAPI")
add_arg_bool_option("dyninst", "TIMEMORY_USE_DYNINST")
add_arg_bool_option("arch", "TIMEMORY_USE_ARCH")
add_arg_bool_option("vtune", "TIMEMORY_USE_VTUNE")
add_arg_bool_option("gperftools", "TIMEMORY_USE_GPERFTOOLS")
add_arg_bool_option("pybind-install", "PYBIND11_INSTALL")
add_arg_bool_option("build-testing", "TIMEMORY_BUILD_TESTING")
parser.add_argument("--cxx-standard", default=14, type=int,
                    choices=[14, 17, 20],
                    help="Set C++ language standard")

args, left = parser.parse_known_args()
# if help was requested, print these options and then add '--help' back
# into arguments so that the skbuild/setuptools argparse catches it
if args.help:
    parser.print_help()
    left.append("--help")
sys.argv = sys.argv[:1] + left

set_cmake_bool_option("TIMEMORY_USE_MPI", args.enable_mpi, args.disable_mpi)
set_cmake_bool_option("TIMEMORY_USE_TAU", args.enable_tau, args.disable_tau)
set_cmake_bool_option("TIMEMORY_USE_CALIPER", args.enable_caliper, args.disable_caliper)
set_cmake_bool_option("TIMEMORY_USE_UPCXX", args.enable_upcxx, args.disable_upcxx)
set_cmake_bool_option("TIMEMORY_USE_CUDA", args.enable_cuda, args.disable_cuda)
set_cmake_bool_option("TIMEMORY_USE_CUPTI", args.enable_cupti, args.disable_cupti)
set_cmake_bool_option("TIMEMORY_USE_PAPI", args.enable_papi, args.disable_papi)
set_cmake_bool_option("TIMEMORY_USE_DYNINST", args.enable_dyninst, args.disable_dyninst)
set_cmake_bool_option("TIMEMORY_USE_ARCH", args.enable_arch, args.disable_arch)
set_cmake_bool_option("TIMEMORY_USE_VTUNE", args.enable_vtune, args.disable_vtune)
set_cmake_bool_option("TIMEMORY_USE_GPERFTOOLS", args.enable_gperftools, args.disable_gperftools)
set_cmake_bool_option("PYBIND11_INSTALL",
                      args.enable_pybind_install, args.disable_pybind_install)
set_cmake_bool_option("TIMEMORY_BUILD_TESTING",
                      args.enable_build_testing, args.disable_build_testing)
cmake_args.append("-DCMAKE_CXX_STANDARD={}".format(args.cxx_standard))


# ---------------------------------------------------------------------------- #
#
def get_project_version():
    # open "VERSION"
    with open(os.path.join(os.getcwd(), 'VERSION'), 'r') as f:
        data = f.read().replace('\n', '')
    # make sure is string
    if isinstance(data, list) or isinstance(data, tuple):
        return data[0]
    else:
        return data


# ---------------------------------------------------------------------------- #
#
def get_long_description():
    long_descript = ''
    try:
        long_descript = open('README.md').read()
    except:
        long_descript = ''
    return long_descript


# ---------------------------------------------------------------------------- #
#
def get_short_description():
    return "{} {} {}".format(
        "Lightweight cross-language instrumentation API for C, C++, Python,",
        "Fortran, and CUDA which allows arbitrarily bundling tools together",
        "into a single performance analysis handle")


# ---------------------------------------------------------------------------- #
def get_keywords():
    return ['timing', 'memory', 'auto-timers', 'signal', 'c++', 'cxx', 'rss',
            'memory', 'cpu time', 'cpu utilization', 'wall clock',
            'system clock', 'user clock', 'pybind11', 'profiling',
            'hardware counters', 'cupti', 'cuda', 'papi', 'caliper',
            'gperftools']


# ---------------------------------------------------------------------------- #
def get_classifiers():
    return [
        # no longer beta
        'Development Status :: 5 - Production/Stable',
        # performance monitoring
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        # can be used for all of below
        'Topic :: Software Development :: Bug Tracking',
        'Topic :: Software Development :: Testing',
        'Topic :: Software Development :: Quality Assurance',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: System :: Logging',
        'Topic :: System :: Monitoring',
        'Topic :: Utilities',
        # written in English
        'Natural Language :: English',
        # MIT license
        'License :: OSI Approved :: MIT License',
        # tested on these OSes
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'Operating System :: POSIX :: Linux',
        'Operating System :: POSIX :: BSD',
        # written in C++, available to Python via PyBind11
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
    ]


# ---------------------------------------------------------------------------- #
def get_name():
    return 'Jonathan R. Madsen'


# ---------------------------------------------------------------------------- #
def get_email():
    return 'jrmadsen@lbl.gov'


# ---------------------------------------------------------------------------- #
class custom_install(skinstall):
    """
    Custom installation
    """

    def __init__(self, *args, **kwargs):
        skinstall.__init__(self, *args, **kwargs)

    def run(self):
        print('\n\n\tRunning install...')
        print('\t\t {:10} : {}'.format("base", self.install_base))
        print('\t\t {:10} : {}'.format("purelib", self.install_purelib))
        print('\t\t {:10} : {}'.format("platlib", self.install_platlib))
        print('\t\t {:10} : {}'.format("headers", self.install_headers))
        print('\t\t {:10} : {}'.format("lib", self.install_lib))
        print('\t\t {:10} : {}'.format("scripts", self.install_scripts))
        print('\t\t {:10} : {}'.format("data", self.install_data))
        print('\t\t {:10} : {}'.format("userbase", self.install_userbase))
        print('\t\t {:10} : {}'.format("usersite", self.install_usersite))
        print('\n\n')
        skinstall.run(self)
        for itr in self.get_outputs():
            print('installed file : "{}"'.format(itr))
        # glob.glob(os.path.join(self.


# ---------------------------------------------------------------------------- #
if platform.system() == "Darwin":
    # scikit-build will set this to 10.6 and C++ compiler check will fail
    version = platform.mac_ver()[0].split('.')
    version = ".".join([version[0], version[1]])
    cmake_args += ["-DCMAKE_OSX_DEPLOYMENT_TARGET={}".format(version)]

# suppress:
#  "setuptools_scm/git.py:68: UserWarning: "/.../tomopy" is shallow and may cause errors"
# since 'error' in output causes CDash to interpret warning as error
with warnings.catch_warnings():
    setup(
        name='timemory',
        packages=['timemory'],
        version=get_project_version(),
        include_package_data=False,
        cmake_args=cmake_args,
        cmake_languages=('C', 'CXX'),
        author=get_name(),
        author_email=get_email(),
        maintainer=get_name(),
        maintainer_email=get_email(),
        contact=get_name(),
        contact_email=get_email(),
        description=get_short_description(),
        long_description=get_long_description(),
        long_description_content_type='text/markdown',
        license='MIT',
        url='http://timemory.readthedocs.io',
        download_url='http://github.com/NERSC/timemory.git',
        zip_safe=False,
        install_requires=[],
        setup_requires=[],
        keywords=get_keywords(),
        classifiers=get_classifiers(),
        python_requires='>=2.7',
        cmdclass=dict(install=custom_install),
        entry_points={
            'console_scripts': ['timemory-plotter=timemory.plotting.__main__:try_plot',
                                'timemory-roofline=timemory.roofline.__main__:try_plot'],
        },
    )
