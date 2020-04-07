#!/bin/bash -e

mkdir -p /opt
cd /opt

git clone https://github.com/spack/spack.git spack
mv /tmp/*.yaml /opt/spack/etc/spack/defaults/

export PATH=/opt/spack/bin:${PATH}

export FORCE_UNSAFE_CONFIGURE=1

apt-get install -y gfortran
FC=$(which gfortran)

spack compiler find
cat /root/.spack/linux/compilers.yaml

ssed -ire 's/f77\:/f77\: \/usr\/bin\/gfortran/g' /root/.spack/linux/compilers.yaml
ssed -ire 's/fc\:/fc\: \/usr\/bin\/gfortran/g' /root/.spack/linux/compilers.yaml

spack compiler find
cat /root/.spack/linux/compilers.yaml

spack install mpich pmi=pmi2
spack install upcxx
spack install gperftools
# spack install likwid
spack install papi +sde
spack install dyninst
spack install caliper +callpath+dyninst+gotcha+libpfm+mpi+sampler ^mpich
spack install tau~ompt+openmp+papi+pdt+pthreads+python+mpi~likwid+libunwind~cuda~phase ^mpich

. /opt/spack/share/spack/setup-env.sh

echo ". /etc/profile.d/modules.sh" > /etc/profile.d/spack.sh
echo ". /opt/spack/share/spack/spack-completion.bash" >> /etc/profile.d/spack.sh
for j in mpi upcxx gperftools papi dyninst caliper tau;
do
    echo "module load $(for i in $(module avail &> /dev/stdout); do echo $i | grep ${j}; done)" >> /etc/profile.d/spack.sh
done

. /etc/profile.d/spack.sh
