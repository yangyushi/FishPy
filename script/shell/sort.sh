#!/usr/bin/env sh
# Move sorted filenames into different sub-folders

frame=$1    # the number of frames belonging to one folder 
interval=$2 # the interval in minutes between two videos (two folders)
filetype=$3 # the type of images, usually tiff

files=($(ls *.$filetype))
count=0

while [[ -n $files ]]; do
    count=$(($count + $interval))
    folder_name=$(printf "min_%04d" $count)
    mkdir $folder_name;
    for ((i=1; i<=$frame; i++)); do
        files=($(ls *.$filetype))
        mv $files $folder_name
    done
    files=($(ls *.$filetype 2> /dev/null)) 
    cd $folder_name
    rename
    cd ..
done
