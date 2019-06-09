#!/usr/bin/env sh
#declare -i n
count=0;

surfix=${1:-"tiff"};
format=${2:-"frame"}
digits=${3:-5}

for filename in *.$surfix;
do
    count=`expr $count + 1`;
    template=%s_%0${digits}d.%s
    new_name=`printf "$template" $format $count $surfix`;
    mv "$filename" "$new_name" 2> /dev/null
done
