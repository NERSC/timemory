# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#
# ----------------------------------------------------------------------------

from spack import *
from sys import platform


class Timemory(CMakePackage):
    """Timing + Memory + Hardware Counter Utilities for C/C++/CUDA/Python"""

    homepage = 'https://timemory.readthedocs.io/en/latest/'
    git = 'https://github.com/NERSC/timemory.git'
    maintainers = ['jrmadsen']

    version('master', branch='master', submodules=True)
    version('develop', branch='develop', submodules=True)
    version('3.0.0', commit='b36b1673b2c6b7ff3126d8261bef0f8f176c7beb',
            submodules=True)
    version('3.0.1', commit='a5bb58b5e4d44b71f699f536ad1b56722f213ce6',
            submodules=True)

    linux = False if platform == 'darwin' else True

    variant('python', default=True, description='Enable Python support')
    variant('mpi', default=True, description='Enable MPI support')
    variant('tau', default=False, description='Enable TAU support')
    variant('papi', default=linux, description='Enable PAPI support')
    variant('ompt', default=True, description='Enable OpenMP tools support')
    variant('cuda', default=linux, description='Enable CUDA support')
    variant('cupti', default=linux, description='Enable CUPTI support')
    variant('tools', default=True, description='Build/install command line tools')
    variant('vtune', default=False, description='Enable VTune support')
    variant('upcxx', default=False, description='Enable UPC++ support')
    variant('gotcha', default=linux, description='Enable GOTCHA support')
    variant('likwid', default=linux, description='Enable LIKWID support')
    variant('caliper', default=True, description='Enable Caliper support')
    variant('dyninst', default=linux,
            description='Enable dynamic instrumentation')
    variant('examples', default=False, description='Build/install examples')
    variant('gperftools', default=True,
            description='Enable gperftools support')
    variant('kokkos_tools', default=True,
            description='Build generic kokkos-tools library, e.g. kp_timemory')
    variant('kokkos_modules', default=False,
            description='Build dedicated kokkos-tools libraries, e.g. kp_timemory_cpu_flops')
    variant('build_caliper', default=True,
            description='Build Caliper submodule instead of spack installation')
    variant('build_gotcha', default=False,
            description='Build GOTCHA submodule instead of spack installation')
    variant('build_ompt', default=True,
            description='Build OpenMP/OMPT submodule instead of spack installation')
    variant('cuda_arch', default='auto', description='CUDA architecture name',
            values=('auto', 'kepler', 'tesla', 'maxwell', 'pascal',
                    'volta', 'turing'), multi=False)

    depends_on('cmake@3.11:', type='build')

    extends('python', when='+python')
    depends_on('python@3:', when='+python', type=('build', 'run'))
    depends_on('py-numpy', when='+python', type=('run'))
    depends_on('py-pillow', when='+python', type=('run'))
    depends_on('py-matplotlib', when='+python', type=('run'))
    depends_on('mpi', when='+mpi')
    depends_on('tau', when='+tau')
    depends_on('papi', when='+papi')
    depends_on('cuda', when='+cuda')
    depends_on('cuda', when='+cupti')
    depends_on('upcxx', when='+upcxx')
    depends_on('likwid', when='+likwid')
    depends_on('gotcha', when='~build_gotcha+gotcha')
    depends_on('caliper', when='~build_caliper+caliper')
    depends_on('dyninst', when='+dyninst')
    depends_on('gperftools', when='+gperftools')
    depends_on('intel-parallel-studio', when='+vtune')

    conflicts('+cupti', when='~cuda', msg='CUPTI requires CUDA')
    conflicts('+kokkos_tools', when='~tools',
              msg='+kokkos_tools requires +tools')
    conflicts('+kokkos_modules', when='~tools',
              msg='+kokkos_modules requires +tools')

    def cmake_args(self):
        spec = self.spec

        # Use spack install of Caliper and/or GOTCHA
        # instead of internal submodule build
        args = [
            '-DTIMEMORY_BUILD_PYTHON=ON',
            '-DTIMEMORY_BUILD_TESTING=OFF',
            '-DTIMEMORY_BUILD_EXTRA_OPTIMIZATIONS=ON',
            '-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON',
        ]

        if '+python' in spec:
            args.append('-DPYTHON_EXECUTABLE={0}'.format(
                spec['python'].command.path))
            args.append('-DTIMEMORY_TLS_MODEL=global-dynamic')

        if '+mpi' in spec:
            args.append('-DMPI_C_COMPILER={0}'.format(spec['mpi'].mpicc))
            args.append('-DMPI_CXX_COMPILER={0}'.format(spec['mpi'].mpicxx))

        if '+cuda' in spec:
            for arch in ('auto', 'kepler', 'tesla', 'maxwell', 'pascal', 'volta',
                         'turing'):
                if "cuda_arch={}".format(arch) in spec:
                    args.append('-DTIMEMORY_CUDA_ARCH={0}'.format(arch))
                    break

        for dep in ('tools', 'examples', 'kokkos_tools'):
            args.append('-DTIMEMORY_BUILD_{}={}'.format(dep.upper(),
                                                        'ON' if '+{}'.format(dep) in spec else 'OFF'))

        for dep in ('build_caliper', 'build_gotcha', 'build_ompt'):
            args.append('-DTIMEMORY_{}={}'.format(dep.upper(),
                                                  'ON' if '+{}'.format(dep) in spec else 'OFF'))

        for dep in ('python', 'mpi', 'tau', 'papi', 'ompt', 'cuda', 'cupti', 'vtune',
                    'upcxx', 'gotcha', 'likwid', 'caliper', 'dyninst', 'gperftools'):
            args.append('-DTIMEMORY_USE_{}={}'.format(dep.upper(),
                                                      'ON' if '+{}'.format(dep) in spec else 'OFF'))

        args.append('-DTIMEMORY_KOKKOS_BUILD_CONFIG={}'.format(
            'ON' if '+kokkos_modules' in spec else 'OFF'))

        return args
