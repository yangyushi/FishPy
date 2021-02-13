#!/usr/bin/env bash

if [ -z "$1" ]; then
    folder="$PWD"
else
    folder="$1"
fi


if cd $folder 
then
    if ls *.mp4 >/dev/null
    then
        echo "processing " $folder
        r1='Basler.*acA2040.*120um.*22673660.*\.mp4'
        r2='Basler.*acA2040.*120um.*22890181.*\.mp4'
        r3='Basler.*acA2040.*55um.*23065350.*\.mp4'
        for file in *.mp4
        do
            if [[ "$file" =~ $r1 ]]; then
                mv "$file" "cam-1.mp4"
            elif [[ "$file" =~ $r2 ]]; then
                mv "$file" "cam-2.mp4"
            elif [[ "$file" =~ $r3 ]]; then
                mv "$file" "cam-3.mp4"
            fi
        done
    fi
else
    exit 1
fi
