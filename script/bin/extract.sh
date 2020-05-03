#!/usr/bin/env bash
# Extract 1st frames of all videos

format="mp4"

if [[ -n $1 ]]; then
    format=$1
fi

for fn in ./*.${format}; do
    echo getting first frame from $fn
    img_name="${fn%.${format}}.png"
    if [[ ! -e $img_name ]]; then
        ffmpeg -y -i ${fn} -vframes 1 ${fn%.${format}}.png  &> /dev/null
    fi
done
