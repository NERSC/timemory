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
    for i in $(find / -type f | egrep -i 'libtimemory|libctimemory')
    do
        echo $(dirname ${i})
    done > ${_TMP}

    cat ${_TMP} | sort -u > /etc/ld.so.conf.d/timemory.conf
}

remove-static-libs()
{
    for i in $@
    do
        find ${i} -type f -regex ".*\.a$" -exec rm -v {} \;
    done
}

remove-broken-links()
{
    for i in $@
    do
        find ${i} -type l ! -exec test -e {} \; -exec echo "  - Removing {}..." \; -exec rm {} \;
    done
}

? ()
{
    awk "BEGIN{ print $* }"
}

calc()
{
    awk "BEGIN{ print $* }"
}
