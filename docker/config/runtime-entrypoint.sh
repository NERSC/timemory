#!/bin/bash -l

set -e
unset PYTHONPATH
export PATH=${PATH}:/opt/local/bin

if [ -z "${1}" ]; then
    exec /bin/bash
    return $?
fi

eval $@
return $?
