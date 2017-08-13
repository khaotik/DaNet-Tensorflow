#!/bin/bash
SAVFILE='saves/timit_1.ckpt'
if [ ! -e $SAVFILE ]; then
    python main.py -o $SAVFILE -tl=64 -ds=timit -ne=10 --no-valid-on-epoch
    if [ ! $? == 0 ]; then exit; fi
    python main.py -ds=timit -i $SAVFILE -m=debug
    python main.py -i $SAVFILE -tl=64 -o $SAVFILE -ds=timit -ne=100 -lr=3e-4 --no-valid-on-epoch
    if [ ! $? == 0 ]; then exit; fi
    python main.py -i $SAVFILE -tl=64 -o $SAVFILE -ds=timit -ne=100 -lr=1e-4 --no-valid-on-epoch
    if [ ! $? == 0 ]; then exit; fi
fi

python main.py -i $SAVFILE -tl=128 -o $SAVFILE -ds=timit -ne=100 -lr=3e-5 --no-valid-on-epoch
if [ ! $? == 0 ]; then exit; fi

python main.py -i $SAVFILE -tl=128 -o $SAVFILE -ds=timit -ne=100 -lr=1e-5 --no-valid-on-epoch
if [ ! $? == 0 ]; then exit; fi

python main.py -i $SAVFILE -tl=256 -o $SAVFILE -ds=timit -ne=100 -lr=3e-6 --no-valid-on-epoch
if [ ! $? == 0 ]; then exit; fi

python main.py -i $SAVFILE -tl=256 -o $SAVFILE -ds=timit -ne=100 -lr=1e-6 --no-valid-on-epoch
