#!/bin/bash -e

EXE=$(basename ${1})
DIR=cpu.prof.${EXE}
mkdir -p ${DIR}

: ${N:=0}
: ${GPERF_PROFILE:=""}
: ${GPERF_PROFILE_BASE:=${DIR}/gperf}
: ${PPROF_ARGS:="--no_strip_temp --lines"}
: ${MALLOCSTATS:=1}
: ${CPUPROFILE_FREQUENCY:=500}
: ${INTERACTIVE:=0}
: ${CPUPROFILE_REALTIME:=1}

while [ -z "${GPERF_PROFILE}" ]
do
    TEST_FILE=${GPERF_PROFILE_BASE}.${N}
    if [ ! -f "${TEST_FILE}" ]; then
        GPERF_PROFILE=${TEST_FILE}
    fi
    N=$((${N}+1))
done

export MALLOCSTATS
export CPUPROFILE_FREQUENCY
export CPUPROFILE_REALTIME

echo -e "\n\t--> Outputting profile to '${GPERF_PROFILE}'...\n"

# remove profile file if unsucessful execution
cleanup-failure() { set +v ; echo "failure"; rm -f ${GPERF_PROFILE}; }
trap cleanup-failure SIGHUP SIGINT SIGQUIT SIGILL SIGABRT SIGKILL

ADD_LIBS()
{
    for i in $@
    do
        if [ -z "${ADD_LIB_LIST}" ]; then
            ADD_LIB_LIST="--add_lib=${i}"
        else
            ADD_LIB_LIST="${ADD_LIB_LIST} --add_lib=${i}"
        fi
    done
}

ADD_PRELOAD()
{
    for i in $@
    do
        if [ -z "${LIBS}" ]; then
            LIBS=${i}
        else
            LIBS="${LIBS}:${i}"
        fi
    done
}

# configure pre-loading of profiler library
PROJECT_LIBRARIES="$(find $PWD | egrep 'libtimemory|libctimemory' | egrep -v '\.a$|\.dSYM')"
ADD_LIBS ${PROJECT_LIBRARIES}
ADD_PRELOAD ${PROJECT_LIBRARIES}
if [ "$(uname)" = "Darwin" ]; then
    ADD_PRELOAD $(otool -L ${1} | egrep 'profiler' | awk '{print $1}')
    LIBS=$(echo ${LIBS} | sed 's/^://g')
    if [ -n "${LIBS}" ]; then
        export DYLD_FORCE_FLAT_NAMESPACE=1
        export DYLD_INSERT_LIBRARIES=${LIBS}
        echo "DYLD_INSERT_LIBRARIES=${DYLD_INSERT_LIBRARIES}"
    fi
else
    ADD_PRELOAD $(ldd ${1} | egrep 'profiler' | awk '{print $(NF-1)}')
    LIBS=$(echo ${LIBS} | sed 's/^://g')
    if [ -n "${LIB}" ]; then
        export LD_PRELOAD=${LIBS}
        echo "LD_PRELOAD=${LD_PRELOAD}"
    fi
fi

# run the application
eval CPUPROFILE_FREQUENCY=${CPUPROFILE_FREQUENCY} CPUPROFILE=${GPERF_PROFILE} $@ | tee ${GPERF_PROFILE}.log

echo-dart-measurement()
{
    local _NAME=${1}
    local _TYPE=${2}
    local _PATH=${3}
    echo "<DartMeasurementFile name=\"${_NAME}\" type=\"image/${_TYPE}\">${_PATH}</DartMeasurementFile>"
}

# generate the results
EXT=so
if [ "$(uname)" = "Darwin" ]; then EXT=dylib; fi
if [ -f "${GPERF_PROFILE}" ]; then
    : ${PPROF:=$(which google-pprof)}
    : ${PPROF:=$(which pprof)}
    if [ -n "${PPROF}" ]; then
        eval ${PPROF} --text ${ADD_LIB_LIST} ${PPROF_ARGS} ${1} ${GPERF_PROFILE} 1> ${GPERF_PROFILE}.txt
        eval ${PPROF} --text --cum ${ADD_LIB_LIST} ${PPROF_ARGS} ${1} ${GPERF_PROFILE} 1> ${GPERF_PROFILE}.cum.txt
        # if dot is available
        if [ -n "$(which dot)" ]; then
            eval ${PPROF} --dot ${ADD_LIB_LIST} ${PPROF_ARGS} ${1} ${GPERF_PROFILE} 1> ${GPERF_PROFILE}.dot
            dot -Tpng ${GPERF_PROFILE}.dot -o ${GPERF_PROFILE}.png
            echo-dart-measurement ${GPERF_PROFILE}.dot png ${PWD}/${GPERF_PROFILE}.png
            if [ -f ./gprof2dot.py ]; then
                eval ${PPROF} --callgrind ${ADD_LIB_LIST} ${PPROF_ARGS} ${1} ${GPERF_PROFILE} 1> ${GPERF_PROFILE}.callgrind
                python ./gprof2dot.py --format=callgrind --output=${GPERF_PROFILE}.callgrind.dot ${GPERF_PROFILE}.callgrind
                dot -Tpng ${GPERF_PROFILE}.callgrind.dot -o ${GPERF_PROFILE}.callgrind.png
                echo-dart-measurement ${GPERF_PROFILE}.callgrind png ${PWD}/${GPERF_PROFILE}.callgrind.png
            fi
        fi
        if [ "${INTERACTIVE}" -gt 0 ]; then
            eval ${PPROF} ${ADD_LIB_LIST} ${PPROF_ARGS} ${1} ${GPERF_PROFILE}
        fi
    else
        echo -e "google-pprof/pprof not found!"
    fi
else
    echo -e "profile file \"${GPERF_PROFILE}\" not found!"
fi
