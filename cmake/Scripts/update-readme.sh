#!/bin/bash

set -o errexit

if [ ${PWD} = ${BASH_SOURCE[0]} ]; then
    cd ../..
fi

pandoc -p --tab-stop=2 --from=markdown --to=rst --output=README.rst README.md
pandoc -p --tab-stop=4 --from=markdown --to=rst --output=CHANGES.rst CHANGES.md

if [ -d ${PWD}/html ]; then
    cp CHANGES.md html/ReleaseNotes.md
fi
