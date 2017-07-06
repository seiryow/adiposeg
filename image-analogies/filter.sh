#!/bin/bash
x=($(ls -1v adiposeg/*.png))
len=${#x[@]}
step=$((len/12))
i=0
while ((i < $len)); do
    echo `realpath ${x[i]}`
    i=$((i + step))
done
