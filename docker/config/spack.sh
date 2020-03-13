#!/bin/bash -e

mkdir -p /opt
cd /opt

git clone https://github.com/spack/spack.git spack

export PATH=/opt/spack/bin:${PATH}

export FORCE_UNSAFE_CONFIGURE=1

spack install dyninst
spack install tau~ompt+openmp+papi+pdt+pthreads+python+mpi~likwid+libunwind~cuda~phase
spack install upcxx
spack install gperftools
spack install ccache
spack install valgrind
spack install papi
# spack install 'intel-parallel-studio@cluster.2020.0'~daal~ipp~mkl+tbb+vtune

. /opt/spack/share/spack/setup-env.sh

echo ". /etc/profile.d/modules.sh" > /etc/profile.d/spack.sh
for j in dyninst tau upcxx gperftools ccache valgrind papi;
do
    echo "module load $(for i in $(module avail &> /dev/stdout); do echo $i | grep ${j}; done)" >> /etc/profile.d/spack.sh
done

. /etc/profile.d/spack.sh
