################################################################################
#   Build stage 1
################################################################################
ARG BASE_IMG=nvidia/cuda
ARG BASE_TAG=latest
FROM ${BASE_IMG}:${BASE_TAG} as builder

USER root
ENV HOME /root
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US
ENV LC_ALL C
ENV SHELL /bin/bash
ENV BASH_ENV /etc/bash.bashrc
ENV DEBIAN_FRONTEND noninteractive

WORKDIR /tmp

# build and env args used by package-manager
ARG COMPILER_TYPE=gcc
ARG GCC_VERSION=8
ARG CLANG_VERSION=9
ARG ENABLE_DISPLAY=1

# package-manager configuration
COPY ./config/apt.sh /tmp/apt.sh
RUN ./apt.sh

# spack configuration
COPY ./config/spack.sh /tmp/spack.sh
RUN ./spack.sh

# build the common dependencies
COPY ./config/timemory-install.sh /tmp/timemory-install.sh
RUN ./timemory-install.sh

# clean up
RUN rm -rf /tmp/*

################################################################################
#   Build stage 2 - compress to 1 layer
################################################################################
FROM scratch

COPY --from=builder / /

ENV HOME /root
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US
ENV LC_ALL C
ENV SHELL /bin/bash
ENV BASH_ENV /etc/bash.bashrc
ENV DEBIAN_FRONTEND noninteractive

ARG REQUIRE_CUDA_VERSION=10.0
ENV CUDA_HOME "/usr/local/cuda"
ENV NVIDIA_REQUIRE_CUDA "cuda>=${REQUIRE_CUDA_VERSION}"
ENV NVIDIA_VISIBLE_DEVICES "all"
ENV NVIDIA_DRIVER_CAPABILITIES "compute,utility"

ENV CC /usr/bin/cc
ENV CXX /usr/bin/c++
ENV CUDACC /usr/local/cuda/bin/nvcc
ENV CUDACXX /usr/local/cuda/bin/nvcc

#------------------------------------------------------------------------------#
#   interactive settings and startup
#------------------------------------------------------------------------------#
COPY ./config/etc/profile.d/*.sh /etc/profile.d/
COPY ./config/etc/bash.bashrc /etc/
COPY ./config/etc/compute-dir-size.py /etc/
COPY ./config/runtime-entrypoint.sh /runtime-entrypoint.sh
COPY ./config/root/* /root/

USER root
WORKDIR /home
SHELL [ "/bin/bash", "--login", "-c" ]
ENTRYPOINT [ "/runtime-entrypoint.sh" ]
