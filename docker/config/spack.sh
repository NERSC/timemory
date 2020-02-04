#!/bin/bash -e

mkdir -p /opt
cd /opt

git clone https://github.com/spack/spack.git spack

export PATH=/opt/spack/bin:${PATH}

export FORCE_UNSAFE_CONFIGURE=1

spack install dyninst

. /opt/spack/share/spack/setup-env.sh

module load $(for i in $(module avail &> /dev/stdout); do echo $i | grep dyninst; done)
module load $(for i in $(module avail &> /dev/stdout); do echo $i | grep boost; done)
