#!/usr/bin/env python

import os
import re
import sys
import sysconfig
import platform
import subprocess

from distutils.version import LooseVersion
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools import Command
from setuptools.command.test import test as TestCommand


# ---------------------------------------------------------------------------- #
class CMakeExtension(Extension):

    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


# ---------------------------------------------------------------------------- #
class CMakeBuild(build_ext, Command):

    use_mpi = 'ON'
    build_type = 'Release'
    cmake_version = '2.7.12'

    def init_cmake(self):
        """
        Ensure cmake is in PATH
        """
        try:
            out = subprocess.check_output(['cmake', '--version'])
            CMakeBuild.cmake_version = LooseVersion(
                re.search(r'version\s*([\d.]+)', out.decode()).group(1))
        except OSError:
            # if fail, try the module
            import cmake
            try:
                if not cmake.CMAKE_BIN_DIR in sys.path:
                    sys.path.append(cmake.CMAKE_BIN_DIR)
                if platform.system() != "Windows":
                    curr_path = os.environ['PATH']
                    if not cmake.CMAKE_BIN_DIR in curr_path:
                        os.environ['PATH'] = "{}:{}".format(curr_path, cmake.CMAKE_BIN_DIR)

                CMakeBuild.cmake_version = cmake.sys.version.split(' ')[0]
            except:
                print ('Error putting cmake in path')
                raise RuntimeError(
                    "CMake must be installed to build the following extensions: " +
                        ", ".join(e.name for e in self.extensions))


    # run
    def run(self):
        self.init_cmake()

        if CMakeBuild.cmake_version < '3.1.3':
            raise RuntimeError("CMake >= 3.1.3 is required")

        print ('Using CMake version {}...'.format(CMakeBuild.cmake_version))

        for ext in self.extensions:
            self.build_extension(ext)


    # build extension
    def build_extension(self, ext):
        self.init_cmake()

        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        mpiarg = '-DUSE_MPI={}'.format(CMakeBuild.use_mpi)
        buildarg = '-DCMAKE_BUILD_TYPE={}'.format(CMakeBuild.build_type)
        cmake_args = ['-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DSETUP_PY=ON',
                      '-DCMAKE_INSTALL_PREFIX=' + extdir,
                      buildarg,
                      mpiarg, ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        install_args = ['--config', cfg]
        if platform.system() == "Windows":
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--target', 'ALL_BUILD', '--', '/m' ]
            install_args += ['--target', 'INSTALL', '--', '/m' ]
        else:
            build_args += [ '--', '-j4' ]
            install_args += [ '--target', 'install' ]

        env = os.environ.copy()
        env['CXXFLAGS'] = '{}'.format(
            env.get('CXXFLAGS', ''))

        # make directory if not exist
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # set to absolute path
        self.build_temp=os.path.abspath(self.build_temp)

        # print the CMake args
        print('CMake args: {}'.format(cmake_args))
        # print the build_args
        print('Build args: {}'.format(build_args))
        # print the install args
        print('Install args: {}'.format(install_args))

        # configure the project
        subprocess.check_call(['cmake'] + cmake_args + [ ext.sourcedir ],
                              cwd=self.build_temp, env=env)

        # build the project
        subprocess.check_call(['cmake', '--build', self.build_temp] + build_args,
                              cwd=self.build_temp, env=env)

        # install the CMake build
        subprocess.check_call(['cmake', '--build', self.build_temp] + install_args,
                              cwd=self.build_temp, env=env)

        print()  # Add an empty line for cleaner output


# ---------------------------------------------------------------------------- #
class CMakeCommand(Command):
    """ Run my command.
    """
    description = 'generate cmake options'

    user_options = [ ('disable-mpi', 's', 'disable building with MPI'),
                     ('debug', 'd', 'debug build'),
                     ('rdebug', 'r', 'release with debug info build'), ]

    def initialize_options(self):
        self.disable_mpi = None
        self.debug = None
        self.rdebug = None
        self.use_mpi = 'ON'
        self.build_type = 'Release'

    def finalize_options(self):
        if self.disable_mpi is not None:
            self.use_mpi = 'OFF'
        if self.rdebug is not None:
            self.build_type = 'RelWithDebInfo'
        if self.debug is not None:
            self.build_type = 'Debug'

    def run(self):
        print ('USE_MPI: {} ({})'.format(self.use_mpi, self.disable_mpi))
        CMakeBuild.use_mpi = self.use_mpi
        CMakeBuild.build_type = self.build_type


# ---------------------------------------------------------------------------- #
class CatchTestCommand(TestCommand):
    """
    A custom test runner to execute both Python unittest tests and C++ Catch-
    lib tests.
    """

    cmake_version = '2.7.12'

    def init_cmake(self):
        """
        Ensure cmake is in PATH
        """
        try:
            out = subprocess.check_output(['cmake', '--version'])

        except OSError:
            # if fail, try the module
            import cmake
            try:
                if not cmake.CMAKE_BIN_DIR in sys.path:
                    sys.path.append(cmake.CMAKE_BIN_DIR)
                if platform.system() != "Windows":
                    curr_path = os.environ['PATH']
                    if not cmake.CMAKE_BIN_DIR in curr_path:
                        os.environ['PATH'] = "{}:{}".format(curr_path, cmake.CMAKE_BIN_DIR)

            except:
                print ('Error putting cmake in path')
                raise RuntimeError(
                    "CMake must be installed to test the following extensions: " +
                        ", ".join(e.name for e in self.extensions))


    def distutils_dir_name(self, dname):
        """Returns the name of a distutils build directory"""
        dir_name = "{dirname}.{platform}-{version[0]}.{version[1]}"
        return dir_name.format(dirname=dname,
                               platform=sysconfig.get_platform(),
                               version=sys.version_info)

    def run(self):
        import cmake
        self.init_cmake()
        print("Running CMake/CTest tests...\n")
        # Run catch tests
        subprocess.call(['ctest'],
                cwd=os.path.join('build', self.distutils_dir_name('temp')), shell=True)


# ---------------------------------------------------------------------------- #
def get_long_description():
    long_descript = ''
    try:
        long_descript = open('README.rst').read()
    except:
        long_descript = ''
    return long_descript


# ---------------------------------------------------------------------------- #
def get_short_description():
    brief_A = 'Python timing (wall, system, user, cpu, %cpu) + RSS memory (current, peak) measurement manager'
    brief_B = 'Written in high-perf C++ and made available to Python via PyBind11'
    return '{}. {}.'.format(brief_A, brief_B)


# ---------------------------------------------------------------------------- #
def get_keywords():
    return [ 'timing', 'memory', 'auto-timers', 'signal', 'c++', 'cxx', 'rss',
             'resident set size', 'cpu time', 'cpu utilization', 'wall clock',
             'system clock', 'user clock', 'pybind11' ]


# ---------------------------------------------------------------------------- #
# calls the setup and declare our 'my_cool_package'
setup(name='TiMemory',
    version='1.1rc0',
    author='Jonathan R. Madsen',
    author_email='jonrobm.programming@gmail.com',
    maintainer='Jonathan R. Madsen',
    maintainer_email='jonrobm.programming@gmail.com',
    contact='Jonathan R. Madsen',
    contact_email='jonrobm.programming@gmail.com',
    description=get_short_description(),
    long_description=get_long_description(),
    url='https://github.com/jrmadsen/TiMemory.git',
    license='MIT',
    # add extension module
    ext_modules=[CMakeExtension('timemory')],
    # add custom build_ext command
    cmdclass=dict(config=CMakeCommand, build_ext=CMakeBuild, test=CatchTestCommand),
    zip_safe=False,
    # extra
    install_requires=[ 'numpy', 'matplotlib', 'argparse', 'cmake', 'mpi4py' ],
    setup_requires=[ 'cmake', 'mpi4py', 'setuptools', 'disttools', 'unittest2' ],
    provides=[ 'timemory' ],
    keywords=get_keywords(),
    python_requires='>=2.6',
)
