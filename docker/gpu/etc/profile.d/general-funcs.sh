#!/bin/bash

cmake-undefine () 
{ 
    for i in $@;
    do
        _tmp=$(grep "^${i}" CMakeCache.txt | grep -v 'ADVANCED' | sed 's/:/ /g' | awk '{print $1}');
        for j in ${_tmp};
        do
            echo "-U${j}";
        done;
    done
}

? ()
{
    awk "BEGIN{ print $* }"
}

calc()
{
    awk "BEGIN{ print $* }"
}
