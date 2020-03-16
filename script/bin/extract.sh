#!/usr/bin/env bash
# Extract 1st frames of all videos
for fn in ./*.mp4; do
    img_name="${fn%.mp4}.png"
    if [[ ! -e $img_name ]]; then
        ffmpeg -y -i ${fn} -vframes 1 ${fn%.mp4}.png  &> /dev/null
    fi
done
