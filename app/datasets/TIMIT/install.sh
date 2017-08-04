#!/bin/bash
unzip TIMIT.zip
if [ ! -f TIMIT.zip ]; then
    echo 'TIMIT.zip not found. Please download it and place it here.'; exit
fi
mkdir train test
for file in `find -type f -name *.WAV | grep TRAIN`; do
    mv $file train/
done

for file in `find -type f -name *.PHN | grep TRAIN`; do
    mv $file train/
done

for file in `find -type f -name *.TXT | grep TRAIN`; do
    mv $file train/
done

for file in `find -type f -name *.WAV | grep TEST`; do
    mv $file test/
done

for file in `find -type f -name *.PHN | grep TEST`; do
    mv $file test/
done

for file in `find -type f -name *.TXT | grep TEST`; do
    mv $file test/
done

cd train/;
for file in `ls | grep .WAV`; do
    sndfile-convert $file ${file,,}
done
rm *.WAV
cd ..;
cd test/;
for file in `ls | grep .WAV`; do
    sndfile-convert $file ${file,,}
done
rm *.WAV
cd ..
rm -rf data/
env python process.py
rm -rf test/
rm -rf train/
echo 'Installation done'
