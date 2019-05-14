#!/bin/bash

set -o errexit

: ${N:=0}
: ${GPERF_PROFILE:=""}
: ${GPERF_PROFILE_BASE:=gperf.heap.prof}
: ${PPROF_ARGS:=""}
: ${MALLOCSTATS:=1}
: ${INTERACTIVE:=0}

while [ -z "${GPERF_PROFILE}" ]
do
    TEST_FILE=${GPERF_PROFILE_BASE}.${N}
    if [ ! -f "${TEST_FILE}" ]; then
        GPERF_PROFILE=${TEST_FILE}
    fi
    N=$((${N}+1))
done

export MALLOCSTATS
echo -e "\n\t--> Outputting profile to '${GPERF_PROFILE}'...\n"

# remove profile file if unsucessful execution
cleanup-failure() { set +v ; rm -f ${GPERF_PROFILE}; }
trap cleanup-failure SIGHUP SIGINT SIGQUIT SIGILL SIGABRT SIGKILL

# configure pre-loading of profiler library
LIBS=$(find $PWD | egrep 'libtimemory|libctimemory' | egrep -v '\.a$|\.dSYM' | sed 's/ /:/g')
if [ "$(uname)" = "Darwin" ]; then
    for i in $(otool -L ${1} | egrep 'tcmalloc|profiler' | awk '{print $1}')
    do
        LIBS=${LIBS}:${i}
    done
    LIBS=$(echo ${LIBS} | sed 's/^://g')
    if [ -n "${LIBS}" ]; then
        export DYLD_FORCE_FLAT_NAMESPACE=1
        export DYLD_INSERT_LIBRARIES=${LIBS}
        echo "DYLD_INSERT_LIBRARIES=${DYLD_INSERT_LIBRARIES}"
    fi
    unset LIBS
else
    for i in $(ldd ${1} | egrep 'tcmalloc|profiler' | awk '{print $(NF-1)}')
    do
        LIBS=${LIBS}:${i}
    done
    LIBS=$(echo ${LIBS} | sed 's/^://g')
    if [ -n "${LIBS}" ]; then
        export LD_PRELOAD=${LIBS}
        echo "LD_PRELOAD=${LD_PRELOAD}"
    fi
    unset LIBS
fi

# run the application
eval HEAPPROFILE=${GPERF_PROFILE} $@ | tee ${GPERF_PROFILE}.log

# generate the results
if [ -f "${GPERF_PROFILE}" ]; then
    : ${PPROF:=$(which pprof)}
    if [ -n "${PPROF}" ]; then
        pprof --text ${PPROF_ARGS} ${1} ${GPERF_PROFILE} &> ${GPERF_PROFILE}.txt
        if [ "${INTERACTIVE}" -gt 0 ]; then
            pprof ${PPROF_ARGS} ${1} ${GPERF_PROFILE}
        fi
    fi
fi
