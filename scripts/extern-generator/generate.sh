#!/bin/bash -e

if [ $# -lt 1 ]; then
    echo "Error! Provide at least one folder name"
    exit 1
fi

for i in $@
do
    echo -e "\n### ${i} ###\n"
    cmake -DFOLDER=${i} -P generate.cmake
done
