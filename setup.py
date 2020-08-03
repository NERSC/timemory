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

cmake_args = ['-DPYTHON_EXECUTABLE={}'.format(sys.executable),
              '-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON',
              '-DTIMEMORY_USE_PYTHON=ON', '-DTIMEMORY_BUILD_PYTHON=ON']
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-h", "--help", help="Print help", action='store_true')

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
    parser.add_argument("--enable-{}".format(lc_name), action='store_true',
                        default=default,
                        help="Explicitly enable {} build".format(disp_name))
    # disable option
    parser.add_argument("--disable-{}".format(lc_name), action='store_true',
                        help="Explicitly disable {} build".format(disp_name))


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

runtime_req_file = ('.requirements/mpi_runtime.txt'
                    if args.enable_mpi and not args.disable_mpi
                    else '.requirements/runtime.txt')

set_cmake_bool_option("TIMEMORY_BUILD_C", args.enable_c, args.disable_c)
set_cmake_bool_option("TIMEMORY_BUILD_TOOLS",
                      args.enable_tools, args.disable_tools)
set_cmake_bool_option("TIMEMORY_USE_MPI", args.enable_mpi, args.disable_mpi)
set_cmake_bool_option("TIMEMORY_USE_NCCL", args.enable_nccl, args.disable_nccl)
set_cmake_bool_option("TIMEMORY_USE_UPCXX", args.enable_upcxx, args.disable_upcxx)
set_cmake_bool_option("TIMEMORY_USE_GOTCHA", args.enable_gotcha,
                      args.disable_gotcha)
set_cmake_bool_option("TIMEMORY_USE_CUDA", args.enable_cuda, args.disable_cuda)
set_cmake_bool_option("TIMEMORY_USE_CUPTI",
                      args.enable_cupti, args.disable_cupti)
set_cmake_bool_option("TIMEMORY_USE_PAPI", args.enable_papi, args.disable_papi)
set_cmake_bool_option("TIMEMORY_USE_ARCH", args.enable_arch, args.disable_arch)
set_cmake_bool_option("TIMEMORY_USE_OMPT", args.enable_ompt, args.disable_ompt)
set_cmake_bool_option("TIMEMORY_USE_DYNINST",
                      args.enable_dyninst, args.disable_dyninst)
set_cmake_bool_option("TIMEMORY_BUILD_OMPT_LIBRARY",
                      args.enable_ompt and args.enable_tools,
                      args.disable_ompt or args.disable_tools)
set_cmake_bool_option("TIMEMORY_BUILD_MPIP_LIBRARY",
                      args.enable_gotcha and args.enable_mpi and args.enable_tools,
                      args.disable_gotcha or args.disable_mpi or args.disable_tools)
set_cmake_bool_option("TIMEMORY_BUILD_NCCLP_LIBRARY",
                      args.enable_gotcha and args.enable_nccl and args.enable_tools,
                      args.disable_gotcha or args.disable_nccl or args.disable_tools)
set_cmake_bool_option("TIMEMORY_BUILD_DYNINST_TOOLS",
                      args.enable_dyninst, args.disable_dyninst)
set_cmake_bool_option("TIMEMORY_BUILD_KOKKOS_TOOLS",
                      args.enable_kokkos, args.disable_kokkos)
set_cmake_bool_option("TIMEMORY_USE_TAU", args.enable_tau, args.disable_tau)
set_cmake_bool_option("TIMEMORY_USE_CALIPER",
                      args.enable_caliper, args.disable_caliper)
set_cmake_bool_option("TIMEMORY_USE_LIKWID", args.enable_likwid,
                      args.disable_likwid)
set_cmake_bool_option("TIMEMORY_USE_VTUNE",
                      args.enable_vtune, args.disable_vtune)
set_cmake_bool_option("TIMEMORY_USE_GPERFTOOLS",
                      args.enable_gperftools, args.disable_gperftools)
set_cmake_bool_option("PYBIND11_INSTALL",
                      args.enable_pybind_install, args.disable_pybind_install)
set_cmake_bool_option("TIMEMORY_BUILD_TESTING",
                      args.enable_build_testing, args.disable_build_testing)
cmake_args.append("-DCMAKE_CXX_STANDARD={}".format(args.cxx_standard))


# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
#
def get_long_description():
    long_descript = ''
    try:
        long_descript = open('README.md').read()
    except Exception:
        long_descript = ''
    return long_descript


# --------------------------------------------------------------------------- #
#
def get_short_description():
    return "{} {} {}".format(
        "Lightweight cross-language instrumentation API for C, C++, Python,",
        "Fortran, and CUDA which allows arbitrarily bundling tools together",
        "into a single performance analysis handle")


# --------------------------------------------------------------------------- #
def get_keywords():
    return ['timing', 'memory', 'auto-timers', 'signal', 'c++', 'cxx', 'rss',
            'memory', 'cpu time', 'cpu utilization', 'wall clock',
            'system clock', 'user clock', 'pybind11', 'profiling',
            'hardware counters', 'cupti', 'cuda', 'papi', 'caliper',
            'gperftools']


# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
def get_name():
    return 'Jonathan R. Madsen'


# --------------------------------------------------------------------------- #
def get_email():
    return 'jrmadsen@lbl.gov'


# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
gotcha_opt = False
if platform.system() == "Darwin":
    # scikit-build will set this to 10.6 and C++ compiler check will fail
    version = platform.mac_ver()[0].split('.')
    version = ".".join([version[0], version[1]])
    cmake_args += ["-DCMAKE_OSX_DEPLOYMENT_TARGET={}".format(version)]
elif platform.system() == "Linux":
    gotcha_opt = True


# --------------------------------------------------------------------------- #
def parse_requirements(fname='requirements.txt', with_version=False):
    """
    Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if true include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
        python -c "import setup; print(chr(10).join(
            setup.parse_requirements(with_version=True)))"
    """
    from os.path import exists
    import re
    require_fpath = fname

    def parse_line(line):
        """
        Parse information from a line in a requirements text file
        """
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(
                            str.strip, rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


# suppress:
#  "setuptools_scm/git.py:68: UserWarning: "/.../<PACKAGE>"
#       is shallow and may cause errors"
# since 'error' in output causes CDash to interpret warning as error
with warnings.catch_warnings():
    print("CMake arguments: {}".format(" ".join(cmake_args)))
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
        setup_requires=[],
        install_requires=parse_requirements(runtime_req_file),
        extras_require={
            'all': parse_requirements('requirements.txt') +
            parse_requirements('.requirements/mpi_runtime.txt'),
            'mpi': parse_requirements('.requirements/mpi_runtime.txt'),
            'build': parse_requirements('.requirements/build.txt'),
        },
        keywords=get_keywords(),
        classifiers=get_classifiers(),
        python_requires='>=3.6',
        cmdclass=dict(install=custom_install),
        entry_points={
            'console_scripts': [
                'timemory-plotter=timemory.plotting.__main__:try_plot',
                'timemory-roofline=timemory.roofline.__main__:try_plot',
                'timemory-python-line-profiler=timemory.line_profiler.__main__:main',
                'timemory-python-profiler=timemory.profiler.__main__:main'],
        },
    )
