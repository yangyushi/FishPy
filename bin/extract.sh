#!/usr/bin/env bash
# Extract a frame from all videos

format="mp4"

if [[ -n $1 ]]; then
    format=$1
fi

delay=0
if [[ -n $2 ]]; then
    delay=$2
fi

for fn in ./*.${format}; do
    echo grabbing frame from ${fn} at ${delay} s
    img_name="${fn%.${format}}.png"
    ffmpeg -ss ${delay} -y -i ${fn} -vframes 1 ${fn%.${format}}.png  &> /dev/null
done
