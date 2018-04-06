#!/bin/bash

set -o errexit

if [ ${PWD} = ${BASH_SOURCE[0]} ]; then
    cd ../..
fi


for i in TiMemory.egg-info build dist .eggs
do
    echo -e "### Removing ${PWD}/${i}..."
    rm -rf ${i}
done
