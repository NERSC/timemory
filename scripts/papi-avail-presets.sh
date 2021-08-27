#!/bin/bash -e

: ${PAPI_AVAIL_EXE:=papi_avail}

AVAIL=$(${PAPI_AVAIL_EXE} -a | grep '^PAPI_' | awk '{print $1}')
MSG=""
for i in ${AVAIL}
do
    if [ -z "${MSG}" ]; then MSG="${i}";
    else MSG="${MSG};${i}"	     
    fi
done
echo "${MSG}"
