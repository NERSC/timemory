#!/bin/bash

set -o errexit

: ${PYTHON_EXE:=python3}

if [ ${PWD} = ${BASH_SOURCE[0]} ]; then
    cd ../../
fi

rm -rf TiMemory.* .eggs build dist
${PYTHON_EXE} setup.py sdist
cd dist
sha256sum *
gpg --detach-sign -a *
# twine upload *

