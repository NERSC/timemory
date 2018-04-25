#!/bin/bash

get-python-version()
{
    local _PYTHON=$(which python)
    if [ -z "${PYTHON}" ]; then _PYTHON=$(which python3); fi
    if [ ! -z "${1}" ]; then _PYTHON=$(which ${1}); fi

    ${_PYTHON} -c "import sys; print('{}.{}'.format(sys.version_info[0], sys.version_info[1]))"
}

write-timemory-ld-config()
{
    local _TMP=$(mktemp /tmp/timemory-XXXX.conf)
    for i in $(find / -type f | grep -i libtimemory)
    do
        echo $(dirname ${i})
    done > ${_TMP}

    cat ${_TMP} | sort -u > /etc/ld.so.conf.d/timemory.conf
}
