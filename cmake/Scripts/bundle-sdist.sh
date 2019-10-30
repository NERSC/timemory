#!/bin/bash

set -o errexit

if [ ${PWD} = ${BASH_SOURCE[0]} ]; then
    cd ../../
fi

rm -rf TiMemory.* .eggs build dist
python setup.py sdist
cd dist
sha256sum *
gpg --detach-sign -a *
# twine upload *

