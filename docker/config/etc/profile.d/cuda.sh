#!/bin/sh

_CUDA=/usr/local/cuda/bin
_NSIGHT_SYSTEMS=/usr/local/cuda/NsightSystems-2019.3/Target-x86_64/x86_64
_NSIGHT_COMPUTE=/usr/local/cuda/NsightCompute-2019.3

PATH=${_CUDA}:${_NSIGHT_SYSTEMS}:${_NSIGHT_COMPUTE}:${PATH}
export PATH

unset _CUDA
unset _NSIGHT_SYSTEMS
unset _NSIGHT_COMPUTE
