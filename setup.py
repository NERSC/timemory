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


# ------------------------------------------------------------------------------------- #
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


# ------------------------------------------------------------------------------------- #
class CMakeBuild(build_ext, Command):

    use_mpi = 'ON'

    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                         out.decode()).group(1))
            if cmake_version < '3.1.3':
                raise RuntimeError("CMake >= 3.1.3 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        mpiarg = '-DUSE_MPI={}'.format(CMakeBuild.use_mpi)
        cmake_args = ['-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DSETUP_PY=ON',
                      '-DCMAKE_INSTALL_PREFIX=' + extdir,
                      mpiarg, ]
        print('CMake args: {}'.format(cmake_args))

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                cfg.upper(),
                extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{}'.format(
            env.get('CXXFLAGS', ''))
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.', '--target', 'install'],
                              cwd=self.build_temp)
        print()  # Add an empty line for cleaner output


# ------------------------------------------------------------------------------------- #
class CMakeCommand(Command):
    """ Run my command.
    """
    description = 'generate cmake options'

    user_options = [ ('disable-mpi', 's', 'disable building with MPI'), ]

    def initialize_options(self):
        self.disable_mpi = None
        self.use_mpi = 'ON'

    def finalize_options(self):
        if self.disable_mpi is not None:
            self.use_mpi = 'OFF'

    def run(self):
        print ('USE_MPI: {} ({})'.format(self.use_mpi, self.disable_mpi))
        CMakeBuild.use_mpi = self.use_mpi


# ------------------------------------------------------------------------------------- #
class CatchTestCommand(TestCommand):
    """
    A custom test runner to execute both Python unittest tests and C++ Catch-
    lib tests.
    """
    def distutils_dir_name(self, dname):
        """Returns the name of a distutils build directory"""
        dir_name = "{dirname}.{platform}-{version[0]}.{version[1]}"
        return dir_name.format(dirname=dname,
                               platform=sysconfig.get_platform(),
                               version=sys.version_info)

    def run(self):
        # Run Python tests
        #super(CatchTestCommand, self).run()
        #print("\nPython tests complete")
        print("Running CMake/CTest tests...\n")
        # Run catch tests
        subprocess.call(['ctest'],
                cwd=os.path.join('build', self.distutils_dir_name('temp')), shell=True)


# ------------------------------------------------------------------------------------- #
def get_long_descript():
    long_descript = ''
    try:
        long_descript = open('README.rst').read()
    except:
        long_descript = ''
    return long_descript


# ------------------------------------------------------------------------------------- #
# calls the setup and declare our 'my_cool_package'
setup(name='TiMemory',
    version='1.0b5',
    author='Jonathan R. Madsen',
    author_email='jonrobm.programming@gmail.com',
    maintainer='Jonathan R. Madsen',
    maintainer_email='jonrobm.programming@gmail.com',
    contact='Jonathan R. Madsen',
    contact_email='jonrobm.programming@gmail.com',
    description='Python timing + memory manager',
    long_description=get_long_descript(),
    url='https://github.com/jrmadsen/TiMemory.git',
    license='MIT',
    # add extension module
    ext_modules=[CMakeExtension('timemory')],
    # add custom build_ext command
    cmdclass=dict(config=CMakeCommand, build_ext=CMakeBuild, test=CatchTestCommand),
    zip_safe=False,
    # extra
    install_requires=[ 'numpy', 'matplotlib', 'argparse', 'cmake' ],
    provides=[ 'timemory' ],
    keywords=[ 'timing', 'memory', 'auto-timers', 'signal', 'c++', 'cxx' ],
    python_requires='>=2.6',
)
