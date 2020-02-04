#!/bin/bash

if [ -d /opt/spack/bin ]; then
    PATH=/opt/spack/bin:${PATH}
    export PATH
fi
