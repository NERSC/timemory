#!/bin/bash

set -o errexit

if [ ${PWD} = ${BASH_SOURCE[0]} ]; then
    cd ../..
fi

pandoc -p --tab-stop=2 --from=markdown --to=rst --output=README.rst README.md
