#!/bin/bash

src_dir=$1
out_dir=$2
num_replica=$3

for file in `find $src_dir -type f`; do
	relpath=`realpath --relative-to=$src_dir $file`
	dst_path="$out_dir/$relpath"
	dst_dir=`dirname $dst_path`
	dst_filename=`basename $dst_path`

	mkdir -p $dst_dir

	for ((prefix = 0; prefix < $num_replica; prefix++)); do
		ln $file $dst_dir/"$prefix"_"$dst_filename"
	done
done
