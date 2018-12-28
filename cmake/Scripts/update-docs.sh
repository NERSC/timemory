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

#------------------------------------------------------------------------------#
# remove old docs in build directory
for i in ${_BINARY}/doc/html/*
do
    rm -rf ${i}
done

#------------------------------------------------------------------------------#
# remove old docs in documentation directory
for i in ${_DOCDIR}/doxy/*
do
    rm -rf ${i}
done

#------------------------------------------------------------------------------#
# switch to build directory
cd ${_BINARY}
# ensure configuration
cmake -DTIMEMORY_DOXYGEN_DOCS=ON -DENABLE_DOXYGEN_HTML_DOCS=ON ${_SOURCE}
# build the docs
cmake --build . --target doc
# copy new docs over
for i in ${_BINARY}/doc/html/*
do
    cp -r ${i} ${_DOCDIR}/doxy/
done

#------------------------------------------------------------------------------#
# go to documentation directory
cd ${_DOCDIR}
# build the documentation
bundle exec jekyll build
