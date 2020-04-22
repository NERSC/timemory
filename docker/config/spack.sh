#!/bin/bash -e

mkdir -p /opt
cd /opt

git clone https://github.com/spack/spack.git spack
mv /tmp/*.yaml /opt/spack/etc/spack/defaults/

NJOBS=4
export PATH=/opt/spack/bin:${PATH}
export FORCE_UNSAFE_CONFIGURE=1
. /opt/spack/share/spack/setup-env.sh

spack compiler find
cat /root/.spack/linux/compilers.yaml

. /opt/spack/share/spack/setup-env.sh

echo ". /etc/profile.d/modules.sh" > /etc/profile.d/spack.sh
echo ". /opt/spack/share/spack/setup-env.sh" >> /etc/profile.d/spack.sh
echo ". /opt/spack/share/spack/spack-completion.bash" >> /etc/profile.d/spack.sh

install-spack-package()
{
    echo -e "\n====> Installing ${1}\n"

    spack spec ${1}
    spack install -j${NJOBS} ${1}
    spack load ${1}
    echo "spack load ${1}" >> /etc/profile.d/spack.sh
}

install-spack-package mpich
install-spack-package papi
install-spack-package dyninst
install-spack-package caliper
install-spack-package gperftools
install-spack-package upcxx

. /etc/profile.d/spack.sh
