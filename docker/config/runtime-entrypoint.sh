#!/bin/bash -l

#set -v
set -e
unset PYTHONPATH

if [ -z "${1}" ]; then
    exec /bin/bash
    return $?
fi

eval $@
return $?
