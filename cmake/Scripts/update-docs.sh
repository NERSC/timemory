#!/bin/bash

set -o errexit

if [ -z "${3}" ]; then
    echo -e "\nPlease provide:"
    echo -e "  (1) build directory"
    echo -e "  (2) source directory"
    echo -e "  (3) html directory\n"
    exit 0
fi

_BINARY=$(realpath ${1})
_SOURCE=$(realpath ${2})
_DOCDIR=$(realpath ${3})

directory-exists()
{
    for i in $@
    do
        if [ ! -d "${i}" ]; then
            echo -e "Directory ${i} does not exist!"
            exit 1
        fi
    done
}

directory-exists "${1}" "${2}" "${3}" "${_BINARY}/doc/html" "${_DOCDIR}/doxy"

DIR=${PWD}

rm -rf ${_BINARY}/doc/html/*
rm -rf ${_DOCDIR}/doxy/*
cd ${_BINARY} && cmake ${_SOURCE} && cmake --build . --target doc
cp -r ${_BINARY}/doc/html/* ${_DOCDIR}/doxy/
cd ${_DOCDIR} && bundle exec jekyll build
