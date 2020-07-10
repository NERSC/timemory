#!/bin/bash -e

if [ $# -lt 1 ]; then
    echo "Error! Provide at least one component folder name"
    exit 1
fi

for i in $@
do
    echo -e "\n### ${i} ###\n"
    cmake -DCOMPONENT_FOLDER=${i} -P generate.cmake
done
