#!/bin/bash

set -e

DIR=$(dirname ${BASH_SOURCE[0]})
: ${BRANCH:=master}
: ${PYTHON:=python3}

if [ "$(uname)" = "Darwin" ]; then
    LOCAL_IP=$(ipconfig getifaddr `route get nersc.gov | grep 'interface:' | awk '{print $NF}'`)
else
    LOCAL_IP=$(ip route ls | tail -n 1 | awk '{print $NF}')
fi

for i in "2018.1" "2018.2"; do
    docker build \
        --tag=timemory:intel_${i} \
        --add-host intel.licenses.nersc.gov:${LOCAL_IP} \
        --build-arg BRANCH=${BRANCH} \
        --build-arg PYTHON=${PYTHON} \
        -f Dockerfile.intel.${i} \
        ${DIR}
done

docker build \
    --tag=timemory:gcc \
    --build-arg BRANCH=${BRANCH} \
    --build-arg PYTHON=${PYTHON} \
    -f Dockerfile.debian \
    ${DIR}
