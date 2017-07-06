#!/bin/bash
echo $#
# sleep 5s
mkdir -p train/label/simple
mkdir -p train/raw/simple
for x in $@; do
    filename=${x##*/}
    basename=${filename%.*}
    # echo $x
    # echo $basename
    make_image_analogy.py "images/cell-A.png" "images/cell-Ap.png" $x "out/cell_$basename"
    echo "======================1"
    cp $x train/label/simple/fake_$basename.png
    echo "======================2"
    cp `ls out/cell_$basename*.png | tail -1` train/raw/simple/fake_$basename.png
    echo "======================3"
done
