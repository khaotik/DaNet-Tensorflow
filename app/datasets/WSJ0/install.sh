#!/bin/bash

# find/install sph2pipe
if [ -e sph2pipe ]; then
    SPH2PIPE='./sph2pipe'
elif command -v sph2pipe > /dev/null; then
    ln -sf `command -v sph2pipe` sph2pipe
    SPH2PIPE='./sph2pipe'
else
    echo "Can't find sph2pipe, installing ..."
    wget "http://www.openslr.org/resources/3/sph2pipe_v2.5.tar.gz"
    if [ $? != 0 ]; then
        echo 'Failed to download sph2pipe, please download and compile it under this directory and try again.'
        exit 1
    fi
    tar -zxf sph2pipe_v2.5.tar.gz
    gcc -o sph2pipe sph2pipe_v2.5/*.c -lm
    if [ $? != 0 ]; then
        echo 'Failed to compile sph2pipe, please put compiled version under this directory and try again.'
        exit 1
    fi
    rm -rvf sph2pipe_v2.5/
    rm sph2pipe_v2.5.tar.gz
fi

# find needed files
find -L -type f -path *si_tr_s*.wv1 > train_set_files
find -L -type f -path *si_dt_05*.wv1 > valid_set_files
find -L -type f -path *si_et_05*.wv1 > test_set_files
python process.py
