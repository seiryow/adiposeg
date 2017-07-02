#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "argument error"
    exit 1
fi

python preprocess.py $1 $2
python train.py $1
python predict.py $1
