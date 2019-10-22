#!/usr/bin/env bash

r1='Basler acA2040-120um \(22673660\)_[0-9]+_[0-9]+\.mp4'
r2='Basler acA2040-120um \(22890181\)_[0-9]+_[0-9]+\.mp4'
r3='Basler acA2040-55um \(23065350\)_[0-9]+_[0-9]+\.mp4'

for file in *.mp4
do
    if [[ "$file" =~ $r1 ]]; then
        mv "$file" "cam-1.mp4"
    elif [[ "$file" =~ $r2 ]]; then
        mv "$file" "cam-2.mp4"
    elif [[ "$file" =~ $r3 ]]; then
        mv "$file" "cam-3.mp4"
    else
        echo $file not a proper video name
    fi
done
