#!/bin/bash -le

# function for running command verbosely
run-verbose()
{
    echo -e "\n\t##### Running : '$@'... #####\n"
    eval $@
}

export PATH=/opt/conda/bin:/usr/local/cuda/bin:${PATH}

export CC=$(which cc)
export CXX=$(which c++)
export CUDACXX=$(which nvcc)
export CPATH=${CPATH}:/usr/include/mpich
export CUDA_HOME=$(realpath /usr/local/cuda)
export LIBRARY_PATH=/usr/local/lib:${LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

ROOT_DIR=${PWD}
: ${TIMEMORY_BRANCH:="master"}

#--------------------------------------------------------------------------------------------#
#                           LIKWID
#--------------------------------------------------------------------------------------------#

run-verbose cd ${ROOT_DIR}
run-verbose git clone https://github.com/RRZE-HPC/likwid.git
run-verbose cd likwid
ssed -i 's/FORTRAN_INTERFACE = false/FORTRAN_INTERFACE = true/g' config.mk
ssed -i 's/NVIDIA_INTERFACE = false/NVIDIA_INTERFACE = true/g' config.mk
ssed -i 's/BUILDAPPDAEMON=false/BUILDAPPDAEMON=true/g' config.mk
run-verbose make -j6
ssed -i 's/@install/install/g' Makefile
ssed -i 's/@cd/cd/g' Makefile
run-verbose make install -j6

#--------------------------------------------------------------------------------------------#
#                           TAU
#--------------------------------------------------------------------------------------------#

# run-verbose cd ${ROOT_DIR}
# run-verbose wget http://tau.uoregon.edu/tau.tgz
# run-verbose tar -xzf tau.tgz
# run-verbose cd tau-*
# export CFLAGS="-O3 -fPIC"
# export CPPFLAGS="-O3 -fPIC"
# run-verbose ./configure -python -prefix=/usr/local -pthread -papi=/usr -mpi -mpiinc=/usr/include/mpich -cuda=/usr/local/cuda
# run-verbose ./configure -python -prefix=/usr/local -pthread -papi=/usr -mpi -mpiinc=/usr/include/mpich
# run-verbose make -j6
# run-verbose make install -j6
# unset CFLAGS
# unset CPPFLAGS

#--------------------------------------------------------------------------------------------#
#                           UPC++
#--------------------------------------------------------------------------------------------#

# run-verbose git clone https://jrmadsen@bitbucket.org/berkeleylab/upcxx.git
# run-verbose cd upcxx
# export CFLAGS="-fPIC"
# export CPPFLAGS="-fPIC"
# run-verbose ./install /usr/local

#--------------------------------------------------------------------------------------------#
#                           timemory
#--------------------------------------------------------------------------------------------#

# run-verbose cd ${ROOT_DIR}
# run-verbose git clone -b ${TIMEMORY_BRANCH} https://github.com/NERSC/timemory.git timemory-source
# run-verbose cd timemory-source

# SOURCE_DIR=$(pwd)
# run-verbose mkdir timemory-build
# run-verbose cd timemory-build

# run-verbose cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=$(which python) -DTIMEMORY_BUILD_C=ON -DTIMEMORY_BUILD_PYTHON=ON ${SOURCE_DIR} -G Ninja
# run-verbose ninja -j6
# run-verbose ninja install

#--------------------------------------------------------------------------------------------#
#                           tomopy
#--------------------------------------------------------------------------------------------#

# run-verbose cd ${ROOT_DIR}
# run-verbose git clone https://github.com/jrmadsen/tomopy.git tomopy
# run-verbose cd tomopy
# run-verbose git checkout accelerated-redesign
# run-verbose conda env create -n tomopy -f envs/linux-36.yml
# source activate
# run-verbose conda activate tomopy
# run-verbose python -m pip install -vvv .
# run-verbose conda clean -a -y

#--------------------------------------------------------------------------------------------#
#                           Cleanup
#--------------------------------------------------------------------------------------------#

run-verbose cd ${ROOT_DIR}
run-verbose rm -rf ${ROOT_DIR}/*
